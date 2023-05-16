from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

# only keep ids of long vm insatnces
def filter_vms(vm_file_path, threshold_minutes = 60):
    vm_table = pd.read_csv(vm_file_path)
    vm_table.columns = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vmtable')]
    vm_long = vm_table[(((vm_table['timestamp vm deleted'] - vm_table['timestamp vm created']) / 300) >= threshold_minutes)]
    return {vm_id: i for i, vm_id in enumerate(vm_long['vm id'])}

# group id and remove duplicates
def process_cpu_reading(path_list, file_count, long_vm_list):
    columns = [content.split(',')[2] for content in open('data/schema.csv').readlines() if content.startswith('vm_cpu_readings')]
    
    df_list = []
    for path in path_list[:file_count]:
        df = pd.read_csv(path)
        df.columns = columns
        df = df.groupby(['timestamp', 'vm id']).mean().reset_index()
        df_filtered = df[df['vm id'].isin(long_vm_list.keys())]
        df_list.append(df_list)

    return df_list

# return one hot vector of vm id
def id_mapper(vm_dict, vm_id):
    return np.eye(vm_dict[vm_id])[vm_id]


if __name__ == '__main__':
    cpu_readings = glob('data/vm_cpu*125.csv')

    df_list = []
    for path in cpu_readings:
        df = pd.read_csv(path, has_header=False)
        df_list.append(df)

    # get list of long vm ids

    # filter out short vm data from cpu reading files

    # save

