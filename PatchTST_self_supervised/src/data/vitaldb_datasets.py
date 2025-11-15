import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm


class EEGSegmentDataset(Dataset):
    def __init__(self, data_dir, segment_sec=20, eeg_rate=128, emg_rate=1, stride_sec=20, mode='pretrain', split='train'):
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
        
        self.split = split
        self.eeg_len = segment_sec * eeg_rate
        self.emg_len = segment_sec * emg_rate
        self.stride_len = stride_sec * eeg_rate
        self.mode = mode  # 'pretrain' or 'finetune'
        self.eeg_rate = eeg_rate
        self.emg_rate = emg_rate
        
        if self.mode == 'pretrain':
            self.create_unsupervised_index()
        else:
            self.create_supervised_index()


    def create_supervised_index(self):
        self.index = [] 

        # self.dist_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}  # to record bis level distribution
        self.dist_dict = {0:0, 1:0, 2:0, 3:0}
        self.bis_list = []
        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        # print(f"Found {len(case_dirs)} cases in {self.data_dir} for mode {self.mode}.")
        # print(case_dirs[:5])
        # print(case_dirs[-5:])
        # case_dirs = case_dirs[:50]
        for case_path in tqdm(case_dirs, desc=f"Reading {self.split} dataset"):
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            emg_path = os.path.join(case_path, 'emg.npy')
            bis_path = os.path.join(case_path, 'bis.npy')
            
            if not (os.path.isfile(eeg_path) and os.path.isfile(emg_path) and os.path.isfile(bis_path)):
                continue
                
            eeg = np.load(eeg_path)# mmap_mode='r')
            eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5)
            emg = np.load(emg_path)# , mmap_mode='r')
            bis = np.load(bis_path)# , mmap_mode='r')

            length = len(eeg)
            for start in range(0, length, self.stride_len):
                eeg_seg = eeg[start:start + self.eeg_len]
                
                start_sec = start // self.eeg_rate
                end_sec = start_sec + self.emg_len
                
                emg_seg = emg[start_sec:end_sec] if end_sec <= len(emg) else emg[start_sec:]
                bis_seg = bis[start_sec:end_sec] if end_sec <= len(bis) else bis[start_sec:]

                if np.isnan(eeg_seg).any() or np.isnan(emg_seg).any() or np.isnan(bis_seg).any():
                    continue

                eeg_seg = self.pad_segment(eeg_seg, self.eeg_len)
                emg_seg = self.pad_segment(emg_seg, self.emg_len)
                bis_seg = self.pad_segment(bis_seg, self.emg_len)

                bis_seg_avg = np.nanmean(bis_seg)
                bis_level = self.get_bis_level(bis_seg_avg)
                emg_seg_avg = np.nanmean(emg_seg)

                self.dist_dict[bis_level] += 1
                self.index.append((eeg_seg, emg_seg_avg, bis_level))
                self.bis_list.append(bis_level)

        return self.index


    def get_label_distribution(self):
        return self.dist_dict
    

    def get_bis_list(self):
        return self.bis_list


    def create_unsupervised_index(self):
        self.index = [] 

        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        for case_path in tqdm(case_dirs, desc=f"Reading {self.split} dataset"):
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            
            if not os.path.isfile(eeg_path):
                continue
                
            eeg = np.load(eeg_path)# mmap_mode='r')
            eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5)
            
            length = len(eeg)
            for start in range(0, length, self.stride_len):
                eeg_seg = eeg[start:start + self.eeg_len]

                if np.isnan(eeg_seg).any():
                    continue
                    
                eeg_seg = self.pad_segment(eeg_seg, self.eeg_len)
                self.index.append(eeg_seg)
        
        return self.index


    def __len__(self):
        return len(self.index)


    def pad_segment(self, segment, target_length):
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)), constant_values=0)
        return segment


    def get_bis_level(self, bis_segment_avg):
        # Reference: 
        # if 80 <= bis_segment_avg <= 100:
        #     return 0  # Class 0: Awake/Alert
        # elif 60 <= bis_segment_avg < 80:
        #     return 1  # Class 1: Light sedation
        # elif 50 <= bis_segment_avg < 60:
        #     return 2  # Class 2: Moderate sedation
        # elif 40 <= bis_segment_avg < 50:
        #     return 3  # Class 3: Moderate sedation
        # elif 25 <= bis_segment_avg < 40:
        #     return 4  # Class 4: Deep anesthesia
        # elif 0 <= bis_segment_avg < 25:
        #     return 5  # Class 5: Deep anesthesia
        # else:
        #     raise ValueError("BIS value out of range [0, 100]")
        if 80 <= bis_segment_avg <= 100:
            return 0  # Class 0: Awake/Alert
        elif 60 <= bis_segment_avg < 80:
            return 1  # Class 1: Light sedation
        elif 40 <= bis_segment_avg < 60:
            return 2  # Class 2: Moderate
        elif 0 <= bis_segment_avg < 40:
            return 3  # Class 3: Deep anesthesia
        else:
            raise ValueError("BIS value out of range [0, 100]")


    def __getitem__(self, idx):
        if self.mode == 'pretrain':
            eeg_segment = self.index[idx]
            eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
            return eeg_tensor, eeg_tensor # first eeg_tensor will be masked in a callback function
        elif (self.mode == 'finetune') or (self.mode == 'alltrain'):
            eeg_seg, emg_avg, bis_level = self.index[idx]

            eeg_tensor = torch.tensor(eeg_seg, dtype=torch.float32)          # (self.eeg_len,)
            emg_avg_tensor = torch.tensor(emg_avg, dtype=torch.float32)     # ()
            bis_level_tensor = torch.tensor(bis_level, dtype=torch.float32)     # ()
            # bis_level_tensor = bis_level_tensor.unsqueeze(0)  # (1,)
            bis_level_tensor = bis_level_tensor.long() # (batch, 1)
            return eeg_tensor, bis_level_tensor
        else:
            raise ValueError("Mode must be 'pretrain' or 'finetune'.")
        

if __name__ == "__main__":
    # dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', mode='pretrain', split='test')
    dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', mode='alltrain', split='test')
    # print(next(iter(dataset)))
    # print(f"Dataset length: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False) # Maybe set num_workers=4
    print(f"Number of batches: {len(dataloader)}")
    for i, (eeg, emg) in enumerate(dataloader):
        print(i, eeg.shape, emg.shape)
