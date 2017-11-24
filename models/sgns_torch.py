from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__MISS__ = 0

from torch.optim import Optimizer


class SparseAdam(Optimizer):
    """Implements lazy version of Adam algorithm suitable for sparse tensors.
    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if not grad.is_sparse:
                    raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg'] = p.data * 0
                    # Exponential moving average of squared gradient values
                    # state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = p.data * 0

                state['step'] += 1

                grad = grad.coalesce()  # the update is non-linear so indices must be unique
                grad_indices = grad._indices()
                grad_values = grad._values()
                size = grad.size()

                def make_sparse(values):
                    constructor = grad.new
                    if grad_indices.dim() == 0 or values.dim() == 0:
                        return constructor().resize_as_(grad)
                    return constructor(grad_indices, values, size)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                #      old <- b * old + (1 - b) * new
                # <==> old += (1 - b) * (new - old)
                old_exp_avg_values = exp_avg._sparse_mask(grad)._values()
                exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                exp_avg.add_(make_sparse(exp_avg_update_values))
                old_exp_avg_sq_values = exp_avg_sq._sparse_mask(grad)._values()
                exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                # Dense addition again is intended, avoiding another _sparse_mask
                numer = exp_avg_update_values.add_(old_exp_avg_values)
                denom = exp_avg_sq_update_values.add_(old_exp_avg_sq_values).sqrt_().add_(group['eps'])
                del exp_avg_update_values, exp_avg_sq_update_values

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(make_sparse(-step_size * numer.div_(denom)))

        return loss


class SGNS(nn.Module):

    def __init__(self, nb_vecs=1000, nb_dims=50):
        super(SGNS, self).__init__()

        self.vecs = nn.Embedding(int(nb_vecs), int(nb_dims), sparse=True)
        self.vecs.weight.data.normal_(0, 0.01)

    def forward(self, vii):
        outputs = Variable(torch.zeros(len(vii)))
        for i in range(len(vii)):
            vecs = self.vecs(vii[i])
            outputs[i] = torch.sum(torch.prod(vecs, dim=0))
        return outputs

    # def forward(self, vii):
    #     vecs = self.vecs(vii)
    #     prod = torch.prod(vecs, dim=1)
    #     return torch.sum(prod, dim=1)

if __name__ == "__main__":

    TRN = pd.read_csv('data/train.csv', usecols=['msno', 'song_id', 'target'], nrows=10000)

    X = np.zeros((len(TRN), 2), dtype=int)
    X[:, 0] = LabelEncoder().fit_transform(TRN['msno'])
    X[:, 1] = LabelEncoder().fit_transform(TRN['song_id']) + 1 + X[:, 0].max()
    y = TRN['target'].values.astype(np.float32)

    print('%d unique users' % len(set(X[:, 0])))
    print('%d unique items' % len(set(X[:, 1])))

    Xt, Xv, ytt, ytv = train_test_split(X, y, test_size=0.3, shuffle=True)

    net = SGNS(nb_vecs=X.max() + 1, nb_dims=100)
    # net.cuda()
    optimizer = SparseAdam(net.parameters(), lr=0.01)
    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 40000
    nb_epochs = 10

    for epoch in range(nb_epochs):
        ii = np.random.permutation(len(Xt))
        loss = 0.
        for i in range(0, len(Xt), batch_size):
            ii_ = ii[i:i + batch_size]
            Xt_ = Xt[ii_]
            ytt_ = ytt[ii_]

            Xt_torch_ = Variable(torch.LongTensor(Xt_))  # .cuda()
            ytt_torch_ = Variable(torch.FloatTensor(ytt_))  # .cuda()

            optimizer.zero_grad()
            ypp_torch_ = net(Xt_torch_)
            loss_ = criterion(ypp_torch_, ytt_torch_)
            loss_.backward()
            optimizer.step()

            loss += loss_.cpu().data.numpy()[0] / (len(Xt) / batch_size)

        print(epoch, loss)
