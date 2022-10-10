import time

import numpy as np
import torch
import torch.nn as nn
from numpy import genfromtxt

from model import TabularModel

start_time = time.time()
torch.manual_seed(33)

# const
CAT_TRAIN = './data/processed/cat_train.csv'
CON_TRAIN = './data/processed/con_train.csv'
Y_TRAIN = './data/processed/y_train.csv'
EMB_SZS = [(24, 12), (2, 1), (7, 4)]
VERSION = '0.0'

# read input data
cat_train = genfromtxt(CAT_TRAIN, delimiter = ',', dtype = np.float)
con_train = genfromtxt(CON_TRAIN, delimiter = ',', dtype = np.float)
y_train = genfromtxt(Y_TRAIN, delimiter = ',', dtype = np.float)

cat_train = torch.tensor(cat_train, dtype=torch.long)
con_train = torch.tensor(con_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)

# print(cat_train.shape)
# print(con_train.shape)
# print(y_train.shape)

# prep the model
model = TabularModel(EMB_SZS, 6, 1, [200, 100], p = 0.4)
# model = TabularModel(EMB_SZS, 6, 1, [5, 3], p = 0.4)    # using smaller model for training

# prep the optimizers and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 300
# epochs = 1
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train))
    losses.append(loss)

    # saving screen space
    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {loss.item(): 10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {loss.item(): 10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# saving the model
torch.save(model.state_dict(), f'./models/model.M.{VERSION}.{int(time.time())}.pt')