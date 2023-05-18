import torch
import torch.nn as nn
import pandas as pd
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset

COLUMN_NAMES = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vm_cpu_readings')]

class TimeSeriesDataset(Dataset):
    def __init__(self, input_window_size, output_window_size, stride = 10, file_count:int = 1):
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.stride = stride
        
        self.vm_table = pd.read_csv('data/vmtable.csv')
        self.cpu_paths = natsorted(glob('data/*125.csv'))[:file_count]
        assert len(self.cpu_paths) == file_count

        # TODO: come up with a way to filter out vm ids if they are only created relatively recently compared to the last timestamp
        

        # filter out vm table using the last timestamp of last file
        valid_ids = None

        # create index dict of one hot vectors for vm and subscription ids
        self.vm_vec = {vm_id: i for i, vm_id in enumerate(self.vm_table['vm id'].values())}
        self.sub_vec = {sub_id: i for i, sub_id in enumerate(self.vm_table['subscription id'].values())}
        self.vm_eye = torch.eye(len(self.vm_vec))
        self.sub_eye = torch.eye(len(self.sub_vec))
        
        # concatenate all cpu readings files, sort by vm id first then timestamp
        self.cpu_concat = pd.concat([self.filter_reading(path, valid_ids) for path in self.cpu_paths], axis = 0).sort_values(['vm id', 'timestamp']) # maybe find a faster way to sort

        # TODO: 

    def filter_reading(path, valid_ids):
        df = pd.read_csv(path)
        df.columns = COLUMN_NAMES
        filtered = df[df['vm id'].isin(valid_ids)]
        return filtered

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pass
        
        



