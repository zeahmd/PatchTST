import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

from scipy.signal import spectrogram


def compute_spectrogram(eeg_signal, fs=128, nperseg=128, noverlap=0):
    f, t, Sxx = spectrogram(eeg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-10)  # Convert to dB scale
    return Sxx


def fill_nan_values(signal_array, sampling_rate, max_gap_seconds=1.0):
    s = pd.Series(signal_array)
    limit_samples = int(max_gap_seconds * sampling_rate)
    cleaned = s.interpolate(method='linear', limit=limit_samples, limit_direction='both')
    return cleaned.to_numpy()


class EEGSegmentDataset(Dataset):
    def __init__(self, data_dir, dataset, segment_sec=20, eeg_rate=128, emg_rate=1, stride_sec=20, use_emg=False, debug=False, mode='pretrain', split='train'):
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
        
        self.dataset = dataset
        assert dataset in ['eeg_time', 'eeg_freq', 'eeg_time_freq'], "Dataset must be one of 'eeg_time', 'eeg_freq', or 'eeg_time_freq'."

        self.split = split
        self.eeg_len = segment_sec * eeg_rate
        self.emg_len = segment_sec * emg_rate
        self.stride_len = stride_sec * eeg_rate
        self.mode = mode  # 'pretrain' or 'finetune'
        self.eeg_rate = eeg_rate
        self.emg_rate = emg_rate
        self.use_emg = use_emg
        self.debug = debug
        
        if self.mode == 'pretrain':
            self.create_unsupervised_index()
        else:
            self.create_supervised_index()


    def create_supervised_index(self):
        self.index = [] 
        self.dist_dict = {0:0, 1:0, 2:0, 3:0}
        self.bis_list = []
        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        # print(f"Found {len(case_dirs)} cases in {self.data_dir} for mode {self.mode}.")
        if self.debug:
            case_dirs = case_dirs[:5]
        remove_count = {
            'any_nan_seg': 0,
            'insufficient_length': 0,
            'spectrogram_nan': 0
        }
        kept_count = 0

        for case_path in tqdm(case_dirs, desc=f"Reading {self.split} dataset"):
            # Step 1: Load EEG, EMG, and BIS data
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            emg_path = os.path.join(case_path, 'emg.npy')
            bis_path = os.path.join(case_path, 'bis.npy')
            if not (os.path.isfile(eeg_path) and os.path.isfile(emg_path) and os.path.isfile(bis_path)): continue
            eeg = np.load(eeg_path)
            emg = np.load(emg_path)
            bis = np.load(bis_path)

            # Step 2: fill 1sec, 2sec gaps in EEG and EMG respectively
            eeg = fill_nan_values(eeg, self.eeg_rate, max_gap_seconds=1.0)
            emg = fill_nan_values(emg, self.emg_rate, max_gap_seconds=2.0)

            # Step 3: Normalize EEG, and EMG
            eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5)
            emg = (emg - np.nanmean(emg)) / (np.nanstd(emg) + 1e-5)
    

            # Step 4: segment data and create index
            length = len(eeg)
            for start in range(0, length, self.stride_len):
                eeg_seg = eeg[start:start + self.eeg_len]
                start_sec = start // self.eeg_rate
                end_sec = start_sec + self.emg_len

                # Step 5: segment EMG, BIS, and Spectrogram
                emg_seg = emg[start_sec:end_sec] if end_sec <= len(emg) else emg[start_sec:]
                bis_seg = bis[start_sec:end_sec] if end_sec <= len(bis) else bis[start_sec:]

                if np.isnan(eeg_seg).any() or np.isnan(emg_seg).any() or np.isnan(bis_seg).any():
                    remove_count['any_nan_seg'] += 1
                    continue
                if len(eeg_seg) < self.eeg_len or len(emg_seg) < self.emg_len or len(bis_seg) < self.emg_len:
                    remove_count['insufficient_length'] += 1
                    continue

                # Step 6: Compute Spectrogram if needed
                if self.dataset in ['eeg_freq', 'eeg_time_freq']:
                    Sxx_seg = compute_spectrogram(eeg_seg, fs=self.eeg_rate, nperseg=128, noverlap=0)
                    if np.isnan(Sxx_seg).any():
                        # print(f"NaN found in spectrogram segment for case {case_path}")
                        remove_count['spectrogram_nan'] += 1
                        continue

                # Step 7: Avg EMG, BIS over segment and get BIS level 
                emg_seg_avg = np.nanmean(emg_seg)
                bis_seg_avg = np.nanmean(bis_seg)
                bis_level = self.get_bis_level(bis_seg_avg)
                self.dist_dict[bis_level] += 1

                # Step 8: Create index based on dataset type
                if self.dataset == 'eeg_time':
                    self.index.append((eeg_seg, emg_seg_avg, bis_level))
                elif self.dataset == 'eeg_freq':
                    self.index.append((Sxx_seg, emg_seg_avg, bis_level))
                elif self.dataset == 'eeg_time_freq':
                        self.index.append((eeg_seg, Sxx_seg, emg_seg_avg, bis_level))
                else:
                    raise ValueError("Dataset must be one of 'eeg_time', 'eeg_freq', or 'eeg_time_freq'.")
                
                kept_count += 1
                self.bis_list.append(bis_level)

        # Step 9: Normalize spectrograms if needed
        if self.dataset in ['eeg_freq', 'eeg_time_freq']:
            if self.dataset == 'eeg_freq':
                all_spect = np.array([self.index[i][0] for i in range(len(self.index))])  # (num_segments, freq_bins, time_bins)
            elif self.dataset == 'eeg_time_freq':
                all_spect = np.array([self.index[i][1] for i in range(len(self.index))])  # (num_segments, freq_bins, time_bins)
            
            mean_spect = np.mean(all_spect)
            std_spect = np.std(all_spect)
            for i in range(len(self.index)):
                if self.dataset == 'eeg_freq':
                    Sxx_seg = self.index[i][0]
                    Sxx_seg = (Sxx_seg - mean_spect) / (std_spect + 1e-5)
                    self.index[i] = (Sxx_seg, self.index[i][1], self.index[i][2])
                elif self.dataset == 'eeg_time_freq':
                    Sxx_seg = self.index[i][1]
                    Sxx_seg = (Sxx_seg - mean_spect) / (std_spect + 1e-5)
                    self.index[i] = (self.index[i][0], Sxx_seg, self.index[i][2], self.index[i][3])

        # Step 10: Print summary
        print(f"Total kept segments: {kept_count}")
        print(f"Total removed segments due to NaN or insufficient length: {remove_count}")
        return self.index


    def get_label_distribution(self):
        return self.dist_dict
    

    def get_bis_list(self):
        return self.bis_list


    def create_unsupervised_index(self):
        self.index = [] 

        case_dirs = sorted(glob(os.path.join(self.data_dir, '*')))
        if self.debug:
            case_dirs = case_dirs[:2]

        remove_count = {
            'any_nan_seg': 0,
            'insufficient_length': 0,
            'spectrogram_nan': 0
        }
        kept_count = 0
        for case_path in tqdm(case_dirs, desc=f"Reading {self.split} dataset"):
            eeg_path = os.path.join(case_path, 'eeg1.npy')
            emg_path = os.path.join(case_path, 'emg.npy')
            if not (os.path.isfile(eeg_path) and os.path.isfile(emg_path)): continue
                
            eeg = np.load(eeg_path)# mmap_mode='r')
            emg = np.load(emg_path)# mmap_mode='r')

            eeg = fill_nan_values(eeg, self.eeg_rate, max_gap_seconds=1.0)
            emg = fill_nan_values(emg, self.emg_rate, max_gap_seconds=2.0)

            eeg = (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-5)
            emg = (emg - np.nanmean(emg)) / (np.nanstd(emg) + 1e-5)

            
            length = len(eeg)
            for start in range(0, length, self.stride_len):
                eeg_seg = eeg[start:start + self.eeg_len]
                start_sec = start // self.eeg_rate
                end_sec = start_sec + self.emg_len

                emg_seg = emg[start_sec:end_sec] if end_sec <= len(emg) else emg[start_sec:]

                if np.isnan(eeg_seg).any() or np.isnan(emg_seg).any():
                    remove_count['any_nan_seg'] += 1
                    continue
                if len(eeg_seg) < self.eeg_len or len(emg_seg) < self.emg_len:
                    remove_count['insufficient_length'] += 1
                    continue
                if self.dataset in ['eeg_freq', 'eeg_time_freq']:
                    Sxx_seg = compute_spectrogram(eeg_seg, fs=self.eeg_rate, nperseg=128, noverlap=0)
                    if np.isnan(Sxx_seg).any():
                        remove_count['spectrogram_nan'] += 1
                        continue

                emg_seg_avg = np.nanmean(emg_seg)

                if self.dataset == 'eeg_time':
                    self.index.append((eeg_seg, emg_seg_avg))
                elif self.dataset == 'eeg_freq':
                    self.index.append((Sxx_seg, emg_seg_avg))
                elif self.dataset == 'eeg_time_freq':
                    self.index.append((eeg_seg, Sxx_seg, emg_seg_avg))
                else:
                    raise ValueError("Dataset must be one of 'eeg_time', 'eeg_freq', or 'eeg_time_freq'.")

                kept_count += 1
                
        if self.dataset in ['eeg_freq', 'eeg_time_freq']:
            if self.dataset == 'eeg_freq':
                all_spect = np.array([self.index[i][0] for i in range(len(self.index))])  # (num_segments, freq_bins, time_bins)
            elif self.dataset == 'eeg_time_freq':
                all_spect = np.array([self.index[i][1] for i in range(len(self.index))])  # (num_segments, freq_bins, time_bins)
            
            mean_spect = np.mean(all_spect)
            std_spect = np.std(all_spect)
            for i in range(len(self.index)):
                if self.dataset == 'eeg_freq':
                    Sxx_seg = self.index[i][0]
                    Sxx_seg = (Sxx_seg - mean_spect) / (std_spect + 1e-5)
                    self.index[i] = (Sxx_seg, self.index[i][1])
                elif self.dataset == 'eeg_time_freq':
                    Sxx_seg = self.index[i][1]
                    Sxx_seg = (Sxx_seg - mean_spect) / (std_spect + 1e-5)
                    self.index[i] = (self.index[i][0], Sxx_seg, self.index[i][2])
        
        print(f"Total kept segments: {kept_count}")
        print(f"Total removed segments due to NaN or insufficient length: {remove_count}")
        return self.index


    def __len__(self):
        return len(self.index)


    def pad_segment(self, segment, target_length):
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)), constant_values=0)
        return segment


    def get_bis_level(self, bis_segment_avg):
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
            if self.dataset == 'eeg_time':
                raise NotImplementedError("Pretrain mode not implemented for 'eeg_time' dataset.")
            elif self.dataset == 'eeg_freq':
                raise NotImplementedError("Pretrain mode not implemented for 'eeg_freq' dataset.")
            elif self.dataset == 'eeg_time_freq':
                eeg_seg, Sxx_seg, emg_avg = self.index[idx]
                eeg_tensor = torch.tensor(eeg_seg, dtype=torch.float32)          # (self.eeg_len,)
                Sxx_tensor = torch.tensor(Sxx_seg, dtype=torch.float32)          # (freq_bins, time_bins)
                # IMPORTANT: Before flattening, change shape: (freq_bins, time_bins) -> (time_bins, freq_bins)
                Sxx_tensor = Sxx_tensor.permute(1, 0)  # (time_bins, freq_bins)
                ##############################################################################################
                Sxx_tensor = Sxx_tensor.flatten()  # flatten time dimension
                emg_avg_tensor = torch.tensor(emg_avg, dtype=torch.float32)     # ()
                return eeg_tensor, Sxx_tensor, emg_avg_tensor
        elif (self.mode == 'finetune') or (self.mode == 'alltrain'):
            if self.dataset == 'eeg_time':
                raise NotImplementedError("Finetune mode not implemented for 'eeg_time' dataset.")
            elif self.dataset == 'eeg_freq':
                raise NotImplementedError("Finetune mode not implemented for 'eeg_freq' dataset.")
            elif self.dataset == 'eeg_time_freq':
                eeg_seg, Sxx_seg, emg_avg, bis_level = self.index[idx]
                eeg_tensor = torch.tensor(eeg_seg, dtype=torch.float32)          # (self.eeg_len,)
                Sxx_tensor = torch.tensor(Sxx_seg, dtype=torch.float32)          
                # IMPORTANT: Before flattening, change shape: (freq_bins, time_bins) -> (time_bins, freq_bins)
                Sxx_tensor = Sxx_tensor.permute(1, 0)  # (time_bins, freq_bins)
                ##############################################################################################
                Sxx_tensor = Sxx_tensor.flatten()  # flatten time dimension
                emg_avg_tensor = torch.tensor(emg_avg, dtype=torch.float32)     
                bis_level_tensor = torch.tensor(bis_level, dtype=torch.float32)
                bis_level_tensor = bis_level_tensor.long() # (batch, 1)
                return eeg_tensor, Sxx_tensor, emg_avg_tensor, bis_level_tensor
        else:
            raise ValueError("Mode must be 'pretrain' or 'finetune'.")
        

if __name__ == "__main__":
    # # dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', mode='pretrain', split='test')
    # # dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', dataset='eeg_freq', mode='alltrain', split='test')
    # dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', dataset='eeg_time', mode='finetune', split='val')
# 
    # # print(next(iter(dataset)))
    # # print(f"Dataset length: {len(dataset)}")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False) # Maybe set num_workers=4
    # print(f"Number of batches: {len(dataloader)}")
    # # for i, (eeg,  emg) in enumerate(dataloader):
    # #     print(i, eeg.shape, emg.shape)
    dataset = EEGSegmentDataset('/home/woody/iwso/iwso204h/vitaldb/cleaned_data', dataset='eeg_time_freq', mode='finetune', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    print(f"Number of batches: {len(dataloader)}")

