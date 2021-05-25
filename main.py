import os
import time
import pandas as pd

import torch
from torch.optim.lr_scheduler import ExponentialLR

from transformers import AdamW 
from mixup_model import kcBERT_custom
from dataloader import make_dataloader
from train import train, evaluate


class Arg:
    batch_size: int = 16
    max_length: int = 150
    train_data_path: str = "C:/Users/장수지/Desktop/jangsj/KoBERTScore/KoBERTScore/train_100.csv"
    val_data_path: str = "C:/Users/장수지/Desktop/jangsj/KoBERTScore/KoBERTScore/train_100.csv"
    test_data_path: str = "C:/Users/장수지/Desktop/jangsj/KoBERTScore/KoBERTScore/train_100.csv"
    lr: float = 5e-6
    epochs: int = 5
    model_save_path: str = 'C:/Users/장수지/Desktop/jangsj/kcc/snapshot'
    model_save_name: str = 'txtclassification.pt'
    device: str = 'cuda:0'


args = Arg()

DEVICE = torch.device(args.device)

train_loader = make_dataloader(args).dataloader(args.train_data_path)
train_mix_loader = make_dataloader(args).dataloader(args.train_data_path, mix='low', DEVICE=DEVICE)
valid_loader = make_dataloader(args).dataloader(args.val_data_path)

model = kcBERT_custom(num_labels = 3, DEVICE=DEVICE)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = ExponentialLR(optimizer, gamma=0.5)


best_val_f1 = None
for e in range(1, args.epochs+1):
    start_time_e = time.time()
    print(f'Model Fitting: [{e}/{args.epochs}]')
    train_loss = train(model, optimizer, scheduler, train_loader, train_mix_loader, DEVICE=DEVICE)
    val_loss, val_f1 = evaluate(model, valid_loader, DEVICE=DEVICE)

    print("[Epoch: %d] train loss : %5.2f | val loss : %5.2f | val f1 score : %5.2f" % (e, train_loss, val_loss, val_f1))
    print(f'Spend Time: [{(time.time() - start_time_e)/60}]')

    # f1 socre가 가장 작은 최적의 모델을 저장
    if not best_val_f1 or val_f1 > best_val_f1:
        if not os.path.isdir(args.model_save_path):
            os.makedirs(args.model_save_path)
        torch.save(model.state_dict(), args.model_save_path + '/' + args.model_save_name)
        print('[save model]')
        best_val_f1 = val_f1
