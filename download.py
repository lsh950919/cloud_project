import os
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def download(url, path = '/scratch/lhk/cpu_readings'):
    file_name = os.path.basename(url)
    print('Downloading:', file_name)
    os.system(f"wget -P {path} {url}")
    os.system(f"gunzip {path}/{file_name}")

def unzip(filename):
    os.system(f'gunzip {filename}')

def split(file_path, train_ids, val_ids, test_ids):
    filename = os.path.basename(file_path)
    print(f'Splitting {filename}')
    filename, extension = filename.split('.')
    df = pd.read_csv(file_path)
    df.columns = CPU_COLUMNS
    
    train_df = df[df['vm id'].isin(train_ids)]
    val_df = df[df['vm id'].isin(val_ids)]
    test_df = df[df['vm id'].isin(test_ids)]

    train_df.to_csv(f'/scratch/lhk/cpu_readings/train/{filename}-train.{extension}', index = False)
    val_df.to_csv(f'/scratch/lhk/cpu_readings/val/{filename}-val{extension}', index = False)
    test_df.to_csv(f'/scratch/lhk/cpu_readings/test/{filename}-test.{extension}', index = False)

    os.remove(file_path)


if __name__ == '__main__':
    with open('./data/links.txt', 'r') as file:
        links = file.readlines()
    numbers = [5, 6, 7, 13, 23, 24, 26, 36, 41, 42, 43, 48]
    links = [f'https://azurecloudpublicdataset.blob.core.windows.net/azurepublicdataset/trace_data/vm_cpu_readings/vm_cpu_readings-file-{number}-of-125.csv.gz\n' for number in numbers]
    
    cpus = cpu_count()
    pool = Pool(cpus)
    for url in tqdm(links):
        pool.apply_async(download, args = (url, ))
    pool.close()
    pool.join()

    cpu_reading_paths = glob('/scratch/lhk/cpu_readings/vm_cpu_readings*.csv')
    vm_table = pd.read_csv('/home/lsh950919/cloud/data/vmtable.csv')
    VM_COLUMNS = [content.split(',')[2] for content in open('/home/lsh950919/cloud/data/schema.csv').readlines() if content.startswith('vmtable')]
    CPU_COLUMNS = [content.split(',')[2] for content in open('/home/lsh950919/cloud/data/schema.csv').readlines() if content.startswith('vm_cpu_readings')]
    vm_table.columns = VM_COLUMNS
    vm_ids = vm_table['vm id']

    train_ids, val_ids = train_test_split(vm_ids, test_size = 0.2)
    val_ids, test_ids = train_test_split(val_ids, test_size = 0.5)
    
    # import ipdb; ipdb.set_trace()
    os.makedirs('/scratch/lhk/cpu_readings/train', exist_ok = True)
    os.makedirs('/scratch/lhk/cpu_readings/val', exist_ok = True)
    os.makedirs('/scratch/lhk/cpu_readings/test', exist_ok = True)

    pool = Pool(cpus)
    for file_path in tqdm(cpu_reading_paths):
        pool.apply_async(split, args = (file_path, train_ids, val_ids, test_ids))
    pool.close()
    pool.join()