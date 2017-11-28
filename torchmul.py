import pdb
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(10)

N = 1000
B = 10

V0 = np.random.randn(N, 20)
V1 = np.random.randn(N, 20)

ii0 = np.random.randint(0, N, (B, 3))
ii1 = np.random.randint(0, N, (B, 5))

V0_ = V0[ii0]
V1_ = V1[ii1]

sums = np.zeros(B)

for b in range(B):
    sums[b] = np.sum(np.matmul(V0_[b], V1_[b].T))

print(sums)


V0_ch = nn.Embedding(N, 20)
V0_ch.weight.data = torch.FloatTensor(V0)
V1_ch = nn.Embedding(N, 20)
V1_ch.weight.data = torch.FloatTensor(V1)

ii0_ch = Variable(torch.LongTensor(ii0))
ii1_ch = Variable(torch.LongTensor(ii1))

V0_ch_ = V0_ch(ii0_ch)
V1_ch_ = V1_ch(ii1_ch)

sums_ch = np.zeros(B)

for b in range(B):
    xx = torch.sum(torch.mm(V0_ch_[b], V1_ch_[b].transpose(1, 0)))
    sums_ch[b] = xx.data.numpy()[0]

print(sums_ch)

muls_ch = torch.matmul(V0_ch_, V1_ch_.transpose(2, 1))
sums_ch = muls_ch.sum(-1).sum(-1)

print(sums_ch)

pdb.set_trace()
