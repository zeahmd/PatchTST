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

        # collect metadata
        self.index = [] 
        # TODO: Maybe discard the cases with more than 30% missing data

        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        for case_path in case_dirs:
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            if not os.path.isfile(eeg_path):
                continue
            eeg = np.load(eeg_path, mmap_mode='r')
            length = len(eeg)
            for start in range(0, length, self.stride_len):
                self.index.append((case_path, start))
        # print(case_dirs)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case_path, start = self.index[idx]

        # Load data
        eeg = np.load(os.path.join(case_path, 'eeg1.npy'))  # high frequency
        emg = np.load(os.path.join(case_path, 'emg.npy'))   # 1 Hz
        bis = np.load(os.path.join(case_path, 'bis.npy'))   # 1 Hz

        # TODO: Perform case-level normalization(e.g., z-score normalization)
        eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5)

        # Interpolate (globally)
        emg = pd.Series(emg).interpolate(method='linear', limit_direction='both').to_numpy()
        bis = pd.Series(bis).interpolate(method='linear', limit_direction='both').to_numpy()

        # Get EEG segment and pad if needed
        eeg_seg = eeg[start:start + self.eeg_len]
        if len(eeg_seg) < self.eeg_len:
            eeg_seg = np.pad(eeg_seg, (0, self.eeg_len - len(eeg_seg)), constant_values=0)

        # Convert eeg sample index to seconds
        start_sec = start // self.eeg_rate
        end_sec = start_sec + self.emg_len

        # Get EMG/BIS segment and pad if needed
        emg_seg = emg[start_sec:end_sec]
        bis_seg = bis[start_sec:end_sec]
        if len(emg_seg) < self.emg_len:
            emg_seg = np.pad(emg_seg, (0, self.emg_len - len(emg_seg)), constant_values=0)
            bis_seg = np.pad(bis_seg, (0, self.emg_len - len(bis_seg)), constant_values=0)

        # Compute targets
        emg_avg = np.nanmean(emg_seg)
        bis_avg = np.nanmean(bis_seg)

        eeg_tensor = torch.tensor(eeg_seg, dtype=torch.float32)          # (640,)
        emg_avg_tensor = torch.tensor(emg_avg, dtype=torch.float32)     # ()
        bis_avg_tensor = torch.tensor(bis_avg, dtype=torch.float32)     # ()

        if self.mode == 'pretrain':
            #  return eeg_tensor, emg_avg_tensor
            return eeg_tensor, eeg_tensor # first eeg_tensor will be masked in a callback function
        elif self.mode == 'finetune':
            return eeg_tensor, bis_avg_tensor
        else:
            raise ValueError("Mode must be 'pretrain' or 'finetune'.")
        

if __name__ == "__main__":
    # dataset = EEGSegmentDataset('/home/permute/pre-train-data', mode='pretrain')
    dataset = EEGSegmentDataset('/home/permute/Documents/FAU Erlangen-Nürnberg/Thesis/data/', mode='pretrain', split='train')
    # print(next(iter(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False) # Maybe set num_workers=4
    for i, (eeg, emg) in enumerate(dataloader):
        print(i, eeg.shape, emg.shape)