# import
import torch
from torch import nn, optim

from network import load_model
from dataset import load_cifar10
from func import train_test, init_weights, unprune_weighs, prune_total_rate, plot_history


# GPU device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu') # 확실히 mps더 몇십배는 더 빠름
model = load_model(device)

############################## hyper parameter ###############################
batch_size = 256
epochs = 150
lr = 3e-4
rr = 5e-4   # regularization rate

lr_factor = 0.9 # lr schedule
lr_patience = 10

prune_term = 70 # prune
prune_rate = 0.2

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=rr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=lr_factor, patience=lr_patience)
###############################################################################

# model init
model.apply(init_weights)
model.apply(unprune_weighs)

# train/test
dl_tr, dl_ts = load_cifar10(batch_size=batch_size)
tr_l, ts_l, tr_a, ts_a = train_test(dl_tr, dl_ts, model, loss_fn, optimizer, scheduler, epochs, prune_term, prune_rate, device)

# plot
print(f"lr: {optimizer.param_groups[0]['lr']}")
print(f"sparsity: {prune_total_rate(model):.2f}%")
plot_history(tr_l, ts_l, tr_a, ts_a,
             lr, rr)