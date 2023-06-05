import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
from natsort import natsorted
from sklearn.preprocessing import MinMaxScaler
import ipdb
import numpy as np
from tqdm import tqdm

random.seed(42)
VM_COLUMNS = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vmtable')]
CPU_COLUMNS = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vm_cpu_readings')]
scaler = MinMaxScaler((0.1, 1))

class TimeSeriesDataset(Dataset):
    def __init__(self, input_window_size, output_window_size, type = 'train', stride = 6, file_count:int = 1, hours:int = 6, scale = False):
        '''
        Args:
            input_window_size (int): number of cpu readings to be used for forecasting
            output_window_size (int): length of forecasting output
            type (str, optional): train, val, or test (default: train)
            stride (int, optional): number of strides in time for each prediction step (default: 10)
            file_count (int, optional): number of cpu files to be used (default: 1)
            hours (int, optional): minimum number of hours vm was used (default: 6)
        '''
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.stride = stride
        
        print('Processing VM table...')
        self.vm_table = pd.read_csv('data/vmtable.csv')
        self.vm_table.columns = VM_COLUMNS
        self.cpu_paths = natsorted(glob('/scratch/lhk/cpu_readings/*125.csv'))[:file_count]
        print(self.cpu_paths)
        print(len(self.cpu_paths))
        assert len(self.cpu_paths) == file_count

        # Encoder Input
        # TODO: come up with a way to filter out vm ids if they are only created relatively recently compared to the last timestamp
        self.vm_table = self.vm_table[(((self.vm_table['timestamp vm deleted'] - self.vm_table['timestamp vm created']) / 300) >= 60*hours)]
        self.vm_table = self.vm_table[['vm id', 'subscription id', 'deployment id', 'vm category', 'vm virtual core count', 'vm memory (gb)']]

        # filter out vm table using the last timestamp of last file
        self.valid_ids = self.vm_table['vm id'].tolist()

        # create index dict of one hot vectors for vm and subscription ids
        self.vm_ind = {vm_id: i for i, vm_id in enumerate(self.vm_table['vm id'].unique())}
        self.sub_ind = {sub_id: i for i, sub_id in enumerate(self.vm_table['subscription id'].unique())}
        self.dep_ind = {dep_id: i for i, dep_id in enumerate(self.vm_table['deployment id'].unique())}
        self.cat_ind = {cat_name: i for i, cat_name in enumerate(self.vm_table['vm category'].unique())}

        self.vm_table['vm id'] = self.vm_table['vm id'].map(self.vm_ind)
        self.vm_table['subscription id'] = self.vm_table['subscription id'].map(self.sub_ind)
        self.vm_table['deployment id'] = self.vm_table['deployment id'].map(self.dep_ind)
        self.vm_table['vm category'] = self.vm_table['vm category'].map(self.cat_ind)

        # scale cores and ram
        if scale:
            self.vm_table['vm virtual core count'] = scaler.fit_transform(self.vm_table[['vm virtual core count']])
            self.vm_table['vm memory (gb)'] = scaler.fit_transform(self.vm_table[['vm memory (gb)']])
        else:
            self.core_ind = {core_count: i for i, core_count in enumerate(self.vm_table['vm virtual core count'].unique())}
            self.ram_ind = {ram: i for i, ram in enumerate(self.vm_table['vm memory (gb)'].unique())}
            self.vm_table['vm virtual core count'] = self.vm_table['vm virtual core count'].map(self.core_ind)
            self.vm_table['vm memory (gb)'] = self.vm_table['vm memory (gb)'].map(self.ram_ind)

        # Decoder Input
        # concatenate all cpu readings files, sort by vm id first then timestamp
        print('Processing CPU reading files...')
        cpu_concat = pd.concat([self.filter_reading(path, self.valid_ids) for path in tqdm(self.cpu_paths)], axis = 0)#.sort_values(['vm id', 'timestamp']) # maybe find a faster way to sort
        cpu_by_time = cpu_concat.pivot_table(index = 'vm id', columns = 'timestamp', values = 'avg cpu').fillna(0)
        self.cpu_by_time = cpu_by_time[[columns for columns in cpu_by_time.columns if columns % 300 == 0]]
    
        cpu_array = np.apply_along_axis(self.window, 1, self.cpu_by_time.to_numpy())
        cpu_label = np.apply_along_axis(self.window, 1, self.cpu_by_time.to_numpy(), label = True)
        self.cpu_data = {vm_index: cpu_array[i] for i, vm_index in enumerate(cpu_by_time.index)}
        self.cpu_label = {vm_index: cpu_label[i] for i, vm_index in enumerate(cpu_by_time.index)}
        self.vm_table = self.vm_table[self.vm_table['vm id'].isin(self.cpu_data.keys())].reset_index(drop = True)

    def window(self, ary, label = False):
        ary_cut = ary[:-((ary.shape[0] - self.stride) % self.input_window_size)]
        if label is False:
            result = np.vstack([ary_cut[i:i+self.input_window_size]
                                for i in range(0, ary_cut.shape[0] - self.input_window_size, self.stride)])
        else:
            result = np.vstack([ary_cut[i+self.input_window_size] 
                            for i in range(0, ary_cut.shape[0] - self.input_window_size, self.stride)])
        return result

    def filter_reading(self, path, valid_ids):
        df = pd.read_csv(path)
        df.columns = CPU_COLUMNS
        filtered = df[df['vm id'].isin(valid_ids)]
        filtered['vm id'] = filtered['vm id'].map(self.vm_ind)
        return filtered
    
    def unique_count(self):
        return len(self.vm_ind), len(self.sub_ind), len(self.dep_ind)
    
    def __len__(self):
        return len(self.vm_table)
    
    def __getitem__(self, index):
        # vm ind, sub ind, dep ind
        vm_id = self.vm_table.loc[index, 'vm id']
        return torch.from_numpy(self.vm_table.loc[index].to_numpy()), \
               torch.from_numpy(self.cpu_data[vm_id]), \
               torch.from_numpy(self.cpu_label[vm_id])
        
if __name__ == '__main__':
    data = TimeSeriesDataset(12, 1, file_count=2)
    # ipdb.set_trace()



