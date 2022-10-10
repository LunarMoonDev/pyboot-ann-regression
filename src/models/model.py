# importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TabularModel(nn.Module):
    '''
        a fully connected ANN with regression output

        @param emb_szs: embedding sizes
        @param n_cont: number of continious features
        @param out_sz: number of output
        @param layers: I/O sizes for each hidden layers
        @param p: dropout ratio
    '''

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5) -> None:
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])       # ModuleList: collates layers
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))                                         # ni = orig size, nf = embed size
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace = True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        '''
            connects the initialized neural layers with the given input

            @param x_cat: categorical inputs
            @param x_cont: continious inputs
        '''
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))                                           # basically each category gets their own embedding table
        x = torch.cat(embeddings, 1)                                                    # concats the embeddings
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x