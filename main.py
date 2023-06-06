import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from src.dataset import TimeSeriesDataset
from src.model import VMEncoder, ForecastDecoder, TimeSeriesTransformer
from src.loss import FocalLoss
import argparse
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
import ipdb
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
min_max_scaler = MinMaxScaler((0, 1))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'encoder')
parser.add_argument('--hidden_dim', type = int, default = 512)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 1024)
args = parser.parse_args()
assert args.mode in ['encoder', 'decoder', 'both'], "Wrong value for mode"


def train_step(data_loader, model, optimizer, scheduler, mode, scaler, cpu_loss, sta_loss, forecast_loss):
    model.train()
    train_loss = 0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    for i, batch in enumerate(data_loader):
        # input, label_cpu, label_sta, cpu_readings, next_cpu = batch # TODO: split batch into label
        # input, label_cpu, label_sta, cpu_readings, next_cpu = input.to(device), label_cpu.to(device), label_sta.to(device), cpu_readings.to(device), next_cpu.to(device)
        input, label_cpu, cpu_readings, next_cpu = batch # TODO: split batch into label
        input, label_cpu, cpu_readings, next_cpu = input.to(device), label_cpu.to(device), cpu_readings.to(device), next_cpu.to(device)
        if input.shape[0] != args.batch_size:
            continue
    
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if mode == 'encoder':
                output = model(input)
                loss_c = cpu_loss(output[0], label_cpu.to(dtype = torch.float32))
                # loss_s = sta_loss(output[1], label_sta)
                # loss = loss_c + loss_s
                # loss_c = cpu_loss(output, label_cpu)
                loss = loss_c
            elif mode == 'decoder':
                output = model(cpu_readings)
                import ipdb; ipdb.set_trace()
                loss_f = forecast_loss(output, label_cpu)
                loss = loss_f
            elif mode == 'both':
                output = model(input, cpu_readings)
                loss_c = cpu_loss(output[0], label_cpu)
                # loss_s = sta_loss(output[1], label_sta)
                loss_f = forecast_loss(output[2], next_cpu) # TODO: label for forecasting
                # loss = loss_c + loss_s + loss_f
                loss = loss_c + loss_f
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        batch_bar.set_postfix(
            loss = f"{train_loss/ (i + 1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        
        batch_bar.update()
    
    batch_bar.close()
    train_loss /= len(data_loader)
    return train_loss

def evaluate(data_loader, model, mode, cpu_loss, sta_loss, forecast_loss):
    model.eval()
    eval_loss = 0
    avg_cpu_correct = 0
    cpu_output = Counter()
    cpu_labels = Counter()
    # stability_output = []
    # forecast_output = []
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')
    
    for i, batch in enumerate(data_loader):
        input, label_cpu, cpu_readings, next_cpu = batch # TODO: split batch into label
        input, label_cpu, cpu_readings, next_cpu = input.to(device), label_cpu.to(device), cpu_readings.to(device), next_cpu.to(device)
        cpu_labels += Counter(label_cpu.clone().detach().cpu().numpy())

        if input.shape[0] != args.batch_size:
            continue
        with torch.no_grad():
            if mode == 'encoder':
                output = model(input)
                loss_c = cpu_loss(output[0], label_cpu)
                # loss_s = sta_loss(output[1], label_sta)
                # loss = loss_c + loss_s
                loss = loss_c
            elif mode == 'decoder':
                loss_f = forecast_loss(output, label_cpu)
                loss = loss_f
            elif mode == 'both':
                output = model(input, cpu_readings)
                loss_c = cpu_loss(output[0], label_cpu)
                # loss_s = sta_loss(output[1], label_sta)
                loss_f = forecast_loss(output[2], next_cpu) # TODO: label for forecasting
                # loss = loss_c + loss_s + loss_f
                loss = loss_c + loss_f
        eval_loss += loss.item()
        avg_cpu_correct += mean_squared_error(output[0].detach().cpu().numpy(), label_cpu.detach().cpu().numpy()) ** 0.5
        # cpu_pred = torch.argmax(F.softmax(output[0], dim = 1), axis = 1)
        # cpu_output += Counter(cpu_pred.clone().detach().cpu().numpy())
        # avg_cpu_correct += int((cpu_pred == label_cpu).sum())
        # stability_correct += int((torch.argmax(F.softmax(output[1]), axis=1) == label_sta).sum())

        batch_bar.set_postfix({'loss': eval_loss / (i + 1)})
        batch_bar.update()
    
    batch_bar.close()
    # cpu_bin_accuracy = avg_cpu_correct / (args.batch_size * len(data_loader)) * 100
    cpu_bin_accuracy = avg_cpu_correct
    # stability_accuracy = stability_correct / batch.shape[0] * len(data_loader)
    
    eval_loss /= len(data_loader)
    return eval_loss, cpu_bin_accuracy, cpu_output, cpu_labels


if __name__ == "__main__":

    # wandb.init(
    # name = "timestep_update",
    # project="cloud",
    # reinit = True,
    # config=args
    # )

    train_dataset = TimeSeriesDataset(12, 1)
    val_dataset = TimeSeriesDataset(12, 1, mode = 'val')
    test_dataset = TimeSeriesDataset(12, 1, mode = 'test')
    vm_unique, sub_unique, dep_unique = train_dataset.unique_count()

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    print([data.shape for data in next(iter(train_loader))])
    if args.mode == 'encoder':
        model = VMEncoder(args.batch_size, vm_unique, sub_unique, dep_unique, args.hidden_dim).to(device)
    elif args.mode == 'decoder':
        model = ForecastDecoder(12, args.hidden_dim).to(device)
    else:
        model = TimeSeriesTransformer(args.batch_size, vm_unique, sub_unique, dep_unique, 12, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 2, T_mult = 2)
    # cpu_loss = torch.nn.CrossEntropyLoss()
    # stability_loss = torch.nn.CrossEntropyLoss()
    # cpu_loss = FocalLoss()
    stability_loss = FocalLoss()
    cpu_loss = torch.nn.MSELoss()
    forecast_loss = torch.nn.MSELoss()

    scaler = GradScaler()
    best_cpu_accuracy = 0
    best_stability_accuracy = 0

    for epoch in range(args.epochs):
    
        epoch_loss = train_step(train_loader, model, optimizer, scheduler, args.mode, scaler, cpu_loss, stability_loss, forecast_loss)
        
        eval_loss, cpu_bin_accuracy, cpu_preds, cpu_labels = evaluate(val_loader, model, args.mode, cpu_loss, stability_loss, forecast_loss)

        scheduler.step()
        print(f'Train loss: {epoch_loss}, Eval loss: {eval_loss}, CPU accuracy: {cpu_bin_accuracy}')
        # wandb.log({'train_loss': float(epoch_loss),
        #            'attention_loss': float(eval_loss),
        #            'CPU accuracy': cpu_bin_accuracy
        #            })
        # print('Preds', cpu_preds)
        # print('Labels', cpu_labels)
        if cpu_bin_accuracy >= best_cpu_accuracy:
            torch.save({'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'val_acc': cpu_bin_accuracy,
            'epoch': epoch}, f'./checkpoints/checkpoint.pth')
            # wandb.save('./checkpoints/checkpoint.pth')

