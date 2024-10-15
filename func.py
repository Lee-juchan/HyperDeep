import torch
from torch import nn
from torch.nn.utils import prune

import matplotlib.pyplot as plt
import time


''' train / test '''

# train loop
def train(dataloader, model, loss_fn, optimizer, scheduler, device):
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    loss_tr = 0
    acc_tr = 0

    model.train() # 훈련모드
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 순전파
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        loss_tr += loss.item()  # item(): tensor 값 추출
        acc_tr += (y_pred.argmax(1) == y).sum().item() / batch_size

        # 역전파
        loss.backward()         # 역전파
        optimizer.step()        # 가중치 조정
        optimizer.zero_grad()   # gradient 초기화 (안하면 중첩됨)

    loss_tr /= num_batches
    acc_tr /= num_batches
    print(f'Tr loss: {loss_tr:.4f}, acc: {acc_tr:.4f}\t', end='')

    return loss_tr, acc_tr

# test loop
def test(dataloader, model, loss_fn, device):
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    loss_ts = 0
    acc_ts = 0
    
    model.eval() # 평가모드
    with torch.no_grad():           # 평가시, gradient 계산하지 않도록
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 순전파
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            loss_ts += loss.item()
            acc_ts += (y_pred.argmax(1) == y).sum().item() / batch_size

    loss_ts /= num_batches
    acc_ts /= num_batches
    print(f"Ts loss: {loss_ts:.4f}, acc: {acc_ts:.4f}")

    return loss_ts, acc_ts

# execute train/test loop
def train_test(dl_tr, dl_ts, model, loss_fn, optimizer, scheduler, epochs, prune_term, prune_rate, device):
    tr_losses = []
    ts_losses = []
    tr_accs = []
    ts_accs = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1:>2d}\t", end='')
        tr_loss, tr_acc = train(dl_tr, model, loss_fn, optimizer, scheduler, device)
        ts_loss, ts_acc = test(dl_ts, model, loss_fn, device)
        
        tr_losses.append(tr_loss)
        ts_losses.append(ts_loss)
        tr_accs.append(tr_acc)
        ts_accs.append(ts_acc)

        # lr schedule
        scheduler.step(ts_loss) # test loss가 일정횟수 이상 감소하는걸 멈추면, lr 감소

        # prune
        # if (epoch != 0) and (epoch % prune_term) == 0:    # 반복 적용
        if epoch == prune_term: # 단일 적용
            model.apply(lambda m : prune_weights(m, amount=prune_rate))

    return tr_losses, ts_losses, tr_accs, ts_accs


''' model '''

# init
def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)

# prune
def prune_weights(module, amount):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, 'weight', amount=amount)
        # prune.ln_structured(module, 'weight', amount=amount, n=2)

def unprune_weighs(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if hasattr(module, 'weight_orig'):  # pruning이 적용된 경우만 존재하는 속성
            prune.remove(module, 'weight')

# prune rate check
def prune_total_rate(model):
    prune_num = 0
    total_num = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune_num += torch.sum(module.weight == 0)
            total_num += module.weight.nelement()
            
    return (prune_num / total_num)

''' plot '''

# train/test history
def plot_history(tr_losses, ts_losses, tr_accs, ts_accs, lr, rr):
    _, ax = plt.subplots(2, 1)

    # loss
    ax[0].plot(tr_losses, label='Train')
    ax[0].plot(ts_losses, label='Test')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # acc
    ax[1].plot(tr_accs, label='Train')
    ax[1].plot(ts_accs, label='Test')
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    t = time.localtime()
    plt.text(0, 0, f'lr={lr}, rr={rr}')
    plt.savefig(f'./graph/{t.tm_year:0>2}{t.tm_mon:0>2}:{t.tm_mday:0>2}{t.tm_hour:0>2}{t.tm_min:0>2}.jpeg')