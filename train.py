import random
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score


def train(model, optimizer, scheduler, train_loader, mix_loader=None, DEVICE=None):
    model.train()
    total_loss = 0
    if mix_loader:
        train_loader = [i for i in train_loader] + [i for i in mix_loader]
        random.shuffle(train_loader)
    for x, y in train_loader:
        optimizer.zero_grad()
        comment, labels = x.to(DEVICE), y.to(DEVICE)
        pred, new_label = model(src_input_sentence=comment, src_label=labels)
        pred = pred.log_softmax(dim=-1)
        loss = torch.mean(torch.sum(-new_label.to(DEVICE) * pred, dim=-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        total_loss = loss.item()

    return total_loss

def evaluate(model, valid_loader, DEVICE=None):
    model.eval()
    val_loss = 0
    val_f1 = 0
    i = 0
    for x, y in valid_loader:
        comment, labels = x.to(DEVICE), y.to(DEVICE)
        pred, _ = model(src_input_sentence=comment, src_label=labels)
        loss = F.cross_entropy(pred, labels)
        val_loss += loss.item()
        val_f1 += f1_score(pred.max(dim=-1)[-1].tolist(), labels.tolist(), average='macro')
        i += 1
    val_loss /= i
    val_f1 /= i
    return val_loss, val_f1
