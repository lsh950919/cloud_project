import os
import numpy as np
from tqdm import tqdm

with open('./links.txt', 'r') as file:
    links = file.readlines()

for url in tqdm(links[36:]):
    try:
        print('Downloading:', url)
        os.system(f'wget {url}')
    
    except KeyboardInterrupt:
        break
