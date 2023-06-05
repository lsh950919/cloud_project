import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split

cpu_reading_paths = glob('/scratch/lhk/cpu_readings/*.csv')

if __name__ == '__main__':
    vm_table = pd.read_csv('data/vmtable.csv')
    VM_COLUMNS = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vmtable')]
    CPU_COLUMNS = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vm_cpu_readings')]

    vm_table.columns = VM_COLUMNS
    vm_ids = vm_table['vm id']

    train_ids, val_ids = train_test_split(vm_ids, test_size = 0.2)
    val_ids, test_ids = train_test_split(val_ids, test_size = 0.5)
    
    # import ipdb; ipdb.set_trace()
    os.makedirs('/scratch/lhk/cpu_readings/train', exist_ok = True)
    os.makedirs('/scratch/lhk/cpu_readings/val', exist_ok = True)
    os.makedirs('/scratch/lhk/cpu_readings/test', exist_ok = True)

    for file_path in cpu_reading_paths:
        filename = os.path.basename(file_path)
        filename, extension = filename.split('.')
        df = pd.read_csv(file_path)
        df.columns = CPU_COLUMNS
        
        train_df = df[df['vm id'].isin(train_ids)]
        val_df = df[df['vm id'].isin(val_ids)]
        test_df = df[df['vm id'].isin(test_ids)]

        train_df.to_csv(f'/scratch/lhk/cpu_readings/train/{filename}-train.{extension}')
        val_df.to_csv(f'/scratch/lhk/cpu_readings/val/{filename}-val{extension}')
        test_df.to_csv(f'/scratch/lhk/cpu_readings/test/{filename}-test.{extension}')

        os.remove(file_path)
        break
