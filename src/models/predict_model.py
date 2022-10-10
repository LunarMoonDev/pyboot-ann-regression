import time

import numpy as np
import torch
import torch.nn as nn
from numpy import genfromtxt

from model import TabularModel

# const
CAT_TEST = './data/processed/cat_test.csv'
CON_TEST = './data/processed/con_test.csv'
Y_TEST = './data/processed/y_test.csv'
MODEL_NAME = 'model.M.0.0.1665313017.pt'
EMB_SZS = [(24, 12), (2, 1), (7, 4)]

# read the test data
cat_test = genfromtxt(CAT_TEST, delimiter = ',', dtype = np.float)
con_test = genfromtxt(CON_TEST, delimiter = ',', dtype = np.float)
y_test = genfromtxt(Y_TEST, delimiter = ',', dtype = np.float)

cat_test = torch.tensor(cat_test, dtype = torch.long)
con_test = torch.tensor(con_test, dtype = torch.float)
y_test = torch.tensor(y_test, dtype = torch.float).reshape(-1, 1    )

# model prep
torch.manual_seed(33)
model = TabularModel(EMB_SZS, 6, 1, [200, 100], p = 0.4)
model.load_state_dict(torch.load(f'./models/{MODEL_NAME}'))
model.eval()

# prep loss func
criterion = nn.MSELoss()

# test the model
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))

print(f'RMSE: {loss: .8f}')

print(f'{"PREDICTED": >12} {"ACTUAL": >8} {"DIFF": >8}')
for i in range(50):
    diff = np.abs(y_val[i].item() - y_test[i].item())
    print(f'{i+2: 2}. {y_val[i].item(): 8.4f} {y_test[i].item(): 8.4f} {diff: 8.4f}')
    