import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from src.dataset import TimeSeriesDataset
from src.model import VMEncoder, ForecastDecoder, TimeSeriesTransformer
import argparse
from torch.cuda.amp import GradScaler
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(data_loader, model, optimizer, scheduler, mode, scaler, cpu_loss, sta_loss, forecast_loss):
    model.train()
    train_loss = 0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    for i, batch in enumerate(data_loader):
        input, label_cpu, label_sta = batch # TODO: split batch into label
        input, label_cpu, label_sta = input.to(device), label_cpu.to(device), label_sta.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input)
            if mode == 'encoder':
                loss_c = cpu_loss(output[0], label_cpu)
                loss_s = sta_loss(output[1], label_sta)
                loss = loss_c + loss_s
            if mode == 'decoder':
                loss_f = forecast_loss(output, label_cpu)
                loss = loss_f
            if mode == 'encoder':
                loss_c = cpu_loss(output[0], label_cpu)
                loss_s = sta_loss(output[1], label_sta)
                loss_s = forecast_loss(output[2], label_sta) # TODO: label for forecasting
                loss = loss_c + loss_s + loss_f
        
        scaler.scaler(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        batch_bar.set_postfix(
            loss = f"{train_loss/ (i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()
    
    batch_bar.close()
    train_loss /= len(data_loader)
    return train_loss

def evaluate():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = 'both')
    parser.add_argument('--hidden_dim', type = int, default = 512)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 100)

    args = parser.parse_args()
    assert args.mode in ['encoder', 'decoder', 'both'], "Wrong value for mode"
    vm_dataset = TimeSeriesDataset(10, 1, mode = args.mode)
    vm_unique, sub_unique, dep_unique = vm_dataset.unique_count()

    train_loader = DataLoader(vm_dataset, batch_size = 512, shuffle = True)
    if args.mode == 'encoder':
        model = VMEncoder(vm_unique, sub_unique, dep_unique, 512)
    elif args.mode == 'decoder':
        model = ForecastDecoder(10, 512)
    else:
        model = TimeSeriesTransformer(vm_unique, sub_unique, dep_unique, 10, 512)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 2, T_mult = 2)
    cpu_loss = torch.nn.CrossEntropyLoss()
    stability_loss = torch.nn.CrossEntropyLoss()
    forecast_loss = torch.nn.MSELoss()

    scaler = GradScaler()

    for epoch in range(args.epochs):
        
        output = train_step(train_loader, model, optimizer, scheduler, args.mode, scaler, cpu_loss, stability_loss, forecast_loss)

