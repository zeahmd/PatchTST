import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from glob import glob


class EEGSegmentDataset(Dataset):
    def __init__(self, data_dir, segment_sec=5, eeg_rate=128, emg_rate=1, stride_sec=5, mode='pretrain', split='train'):
        self.data_dir = data_dir + '/'
        if (split == 'train' and mode == 'pretrain'):
            self.data_dir = os.path.join(self.data_dir, 'pre-train')
        elif (split == 'train' and mode == 'alltrain'):
            self.data_dir = os.path.join(self.data_dir, 'all-train')
        elif (split == 'train' and mode == 'finetune'):
            self.data_dir = os.path.join(self.data_dir, 'fine-train')
        elif split == 'val':
            self.data_dir = os.path.join(self.data_dir, 'val')
        elif split == 'test':
            self.data_dir = os.path.join(self.data_dir, 'test')
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")
        self.eeg_len = segment_sec * eeg_rate
        self.emg_len = segment_sec * emg_rate
        self.stride_len = stride_sec * eeg_rate
        self.mode = mode  # 'pretrain' or 'finetune'
        self.eeg_rate = eeg_rate
        self.emg_rate = emg_rate
        
        self.create_index() # list of (case_path, start_idx) tuples

    def create_index(self):
        self.index = [] 

        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        for case_path in case_dirs:
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            emg_path = os.path.join(case_path, 'emg.npy')
            bis_path = os.path.join(case_path, 'bis.npy')
            
            # Check if all required files exist
            if not (os.path.isfile(eeg_path) and os.path.isfile(emg_path) and os.path.isfile(bis_path)):
                continue
                
            eeg = np.load(eeg_path, mmap_mode='r')
            emg = np.load(emg_path, mmap_mode='r')
            bis = np.load(bis_path, mmap_mode='r')

            # ############ Debugging ############
            # eeg = eeg[1000*128:2000*128]  # 1000 seconds of data
            # emg = emg[1000*1:2000*1]
            # bis = bis[1000*1:2000*1]
            # ###################################
            
            length = len(eeg)
            for start in range(0, length, self.stride_len):
                eeg_seg = eeg[start:start + self.eeg_len]
                
                start_sec = start // self.eeg_rate
                end_sec = start_sec + self.emg_len
                
                emg_seg = emg[start_sec:end_sec] if end_sec <= len(emg) else emg[start_sec:]
                bis_seg = bis[start_sec:end_sec] if end_sec <= len(bis) else bis[start_sec:]

                if np.isnan(eeg_seg).any() or np.isnan(emg_seg).any() or np.isnan(bis_seg).any():
                    continue
                    
                self.index.append((case_path, start))
        
        return self.index

    def __len__(self):
        return len(self.index)
    
    def pad_segment(self, segment, target_length):
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)), constant_values=0)
        return segment

    def get_bis_level(self, bis_segment_avg):
        # Reference: 
        if 80 <= bis_segment_avg <= 100:
            return 0  # Class 0: Awake/Alert
        elif 60 <= bis_segment_avg < 80:
            return 1  # Class 1: Light sedation
        elif 50 <= bis_segment_avg < 60:
            return 2  # Class 2: Moderate sedation
        elif 40 <= bis_segment_avg < 50:
            return 3  # Class 3: Deep sedation
        elif 25 <= bis_segment_avg < 40:
            return 4  # Class 4: General anesthesia
        elif 0 <= bis_segment_avg < 25:
            return 5  # Class 5: Deep anesthesia
        else:
            raise ValueError("BIS value out of range [0, 100]")

    def __getitem__(self, idx):
        case_path, start = self.index[idx]

        # Load data
        eeg = np.load(os.path.join(case_path, 'eeg1.npy'))  # high frequency
        emg = np.load(os.path.join(case_path, 'emg.npy'))   # 1 Hz
        bis = np.load(os.path.join(case_path, 'bis.npy'))   # 1 Hz
        eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5) # case-level normalization(e.g., z-score normalization)
        
        eeg_seg = eeg[start:start + self.eeg_len]
        eeg_seg = self.pad_segment(eeg_seg, self.eeg_len)

        start_sec = start // self.eeg_rate
        end_sec = start_sec + self.emg_len
        emg_seg = emg[start_sec:end_sec]
        bis_seg = bis[start_sec:end_sec]
        emg_seg = self.pad_segment(emg_seg, self.emg_len)
        bis_seg = self.pad_segment(bis_seg, self.emg_len)
        emg_avg = np.nanmean(emg_seg)
        bis_avg = np.nanmean(bis_seg)
        bis_level = self.get_bis_level(bis_avg)
        # bis_level = bis_avg

        eeg_tensor = torch.tensor(eeg_seg, dtype=torch.float32)          # (640,)
        emg_avg_tensor = torch.tensor(emg_avg, dtype=torch.float32)     # ()
        bis_level_tensor = torch.tensor(bis_level, dtype=torch.float32)     # ()
        # bis_level_tensor = bis_level_tensor.unsqueeze(0)  # (1,)
        bis_level_tensor = bis_level_tensor.long() # (batch, 1)

        if self.mode == 'pretrain':
            #  return eeg_tensor, emg_avg_tensor
            return eeg_tensor, eeg_tensor # first eeg_tensor will be masked in a callback function
        elif (self.mode == 'finetune') or (self.mode == 'alltrain'):
            return eeg_tensor, bis_level_tensor
        else:
            raise ValueError("Mode must be 'pretrain' or 'finetune'.")
        

if __name__ == "__main__":
    # dataset = EEGSegmentDataset('/home/permute/pre-train-data', mode='pretrain')
    dataset = EEGSegmentDataset('/home/permute/Documents/FAU Erlangen-Nürnberg/Thesis/data/', mode='alltrain', split='train')
    # print(next(iter(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False) # Maybe set num_workers=4
    for i, (eeg, emg) in enumerate(dataloader):
        print(i, eeg.shape, emg.shape)