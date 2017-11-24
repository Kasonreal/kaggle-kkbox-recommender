from itertools import product
from math import log, ceil
from more_itertools import flatten
from os.path import exists
from os import getenv
from sklearn.metrics import roc_auc_score
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pickle
import pdb
import sys

from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.optim as optim

NTRN = 7377418
NTST = 2556790
np.random.seed(865)

# Reserved index for missing values.
MISSING = 'MISSING'

assert getenv('CUDA_VISIBLE_DEVICES') is not None, "Specify a GPU"
assert len(getenv('CUDA_VISIBLE_DEVICES')) > 0, "Specify a GPU"


class SparseAdam(torch.optim.Optimizer):
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


class GenericVectorSpace(nn.Module):

    def __init__(self,
                 feats_unique,
                 vec_size=50,
                 vec_init_func=np.random.normal,
                 vec_init_params=(0, 0.01),
                 optimizer=SparseAdam,
                 optimizer_params={'lr': 0.01}):
        super(GenericVectorSpace, self).__init__()

        # Lookup feature tuple -> index.
        # TODO: lookup for missing features.
        self.feat2idx = dict(zip(feats_unique, list(range(len(feats_unique)))))

        # Single embedding for all features.
        self.vecs = nn.Embedding(len(feats_unique), vec_size, sparse=True)

        # Initialize weights using given random distribution and parameters.
        vw = vec_init_func(*vec_init_params, (len(feats_unique), vec_size))
        self.vecs.weight.data = torch.FloatTensor(vw.astype(np.float32))

        # Training criteria.
        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def fit_batch(self, X, yt):

        # Convert features in X to vector indexes.
        # 0.0005 seconds for 630 rows.
        X_idxs = [[self.feat2idx[x1], self.feat2idx[x2]] for x1, x2 in X]

        # Convert given variables.
        X_idxs_torch = Variable(torch.LongTensor(X_idxs)).cuda()
        yt_torch = Variable(torch.FloatTensor(yt)).cuda()

        # Forward pass.
        self.optimizer.zero_grad()
        yp_torch = self.forward(X_idxs_torch)
        loss = self.criterion(yp_torch, yt_torch)

        # Backward pass and update.
        loss.backward()
        self.optimizer.step()

        # Compute, return metrics.
        loss = loss.cpu().data.numpy()[0]
        yp = self._sigmoid(yp_torch.cpu().data.numpy())
        auc = roc_auc_score(yt, yp)
        return loss, auc

    # def fit_generator(self, batch_gen, batch_size, nb_samples, stop_func):
    #     nb_epochs = 0
    #     return

    def predict_batch(self, X):
        # TODO: apply sigmoid after forward() call.
        return

    def forward(self, X_idxs):

        # Retrieve the feature vectors by their indexes.
        X_vecs = self.vecs(X_idxs)

        # Element-wise product across the sample dimension. (b, 2, v) -> (b, v)
        prod = torch.prod(X_vecs, dim=1)

        # Sum across the sample dimension. (b, v) -> (b,)
        return torch.sum(prod, dim=1)


def round_to(n, r):
    return round(n / r) * r


def get_user_feats(df):
    """For each user, create a mapping from a unique user key to
    a list of [feature type, feature value] pairs."""

    # Key is the user id prefixed by "u-".
    msno2key = lambda x: 'u-%s' % x
    keys = df['msno'].apply(msno2key).values.tolist()

    # Remove duplicate based on Id.
    df = df.drop_duplicates('msno')
    keys_dedup = df['msno'].apply(msno2key).values.tolist()

    # Build mapping from unique keys to features.
    keys2feats = {k: [] for k in keys_dedup}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        # User msno (id).
        keys2feats[k].append(('u-msno', row['msno']))

        # User age. Clipped and rounded.
        if not (0 < row['bd'] < 70):
            keys2feats[k].append(('u-age', MISSING))
        else:
            keys2feats[k].append(('u-age', round_to(row['bd'], 5)))

        # User city. No missing values.
        keys2feats[k].append(('u-city', int(row['city'])))

        # User gender. Missing if not female or male.
        if row['gender'] not in {'male', 'female'}:
            keys2feats[k].append(('u-sex', MISSING))
        else:
            keys2feats[k].append(('u-sex', row['gender']))

        # User registration method. No missing values.
        keys2feats[k].append(('u-reg-via', int(row['registered_via'])))

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        keys2feats[k].append(('u-reg-year', y0))

        # User account age, in years. No missing values.
        y1 = int(str(row['expiration_date'])[:4])
        keys2feats[k].append(('u-act-age', y1 - y0))

    return keys, keys2feats


def get_song_feats(df):

    # Key is the song id prefixed by "s-"
    songid2key = lambda x: 's-%s' % x
    keys = df['song_id'].apply(songid2key).values.tolist()

    # Remove duplicates.
    df = df.drop_duplicates('song_id')
    keys_dedup = df['song_id'].apply(songid2key).values.tolist()

    # Build mapping from unique keys to features.
    keys2feats = {k: [] for k in keys_dedup}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        # Song id.
        keys2feats[k].append(('s-id', row['song_id']))

        # Song length. Missing if nan.
        if np.isnan(row['song_length']):
            keys2feats[k].append(('s-len', MISSING))
        else:
            f = row['song_length'] / 1000 / 60
            f = round(log(max(1, f)))
            keys2feats[k].append(('s-len', f))

        # Song language. Missing if nan.
        if np.isnan(row['language']):
            keys2feats[k].append(('s-lang', MISSING))
        else:
            keys2feats[k].append(('s-lang', int(row['language'])))

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is not str:
            keys2feats[k].append(('s-year', MISSING))
            keys2feats[k].append(('s-country', MISSING))
        else:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            keys2feats[k].append(('s-year', f))
            keys2feats[k].append(('s-country', row['isrc'][:2]))

        # Song genre(s). Missing if nan. Split on pipes.
        if type(row['genre_ids']) is not str:
            keys2feats[k].append(('s-genre', MISSING))
        else:
            for g in row['genre_ids'].split('|'):
                keys2feats[k].append(('s-genre', int(g)))

        # Song musicians. Combine artist, composer, lyricist.
        if str not in {type(row['artist_name']), type(row['composer']), type('lyricist')}:
            keys2feats[k].append('s-musician', MISSING)

        # Artist. Missing if nan, but already accounted for.
        if type(row['artist_name']) == str:
            for m in row['artist_name'].split('|'):
                keys2feats[k].append(('s-musician', m.strip()))

        # Composer. Missing if nan, but already accounted for.
        if type(row['composer']) == str:
            for m in row['composer'].split('|'):
                keys2feats[k].append(('s-musician', m.strip()))

        # Lyricist. Missing if nan, but already accounted for.
        if type(row['lyricist']) == str:
            for m in row['lyricist'].split('|'):
                keys2feats[k].append(('s-musician', m.strip()))

    return keys, keys2feats


class MultiVecRec(object):

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 best_model_path,
                 predict_path):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.best_model_path = best_model_path
        self.predict_path = predict_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_features(self, train=False, test=False):

        path_sample_keys_trn = '%s/data-sample-keys-trn.csv' % self.artifacts_dir
        path_sample_keys_tst = '%s/data-sample-keys-tst.csv' % self.artifacts_dir
        path_feats = '%s/data-feats.pkl' % self.artifacts_dir

        pp = [path_sample_keys_trn, path_sample_keys_tst, path_feats]
        feats_ready = sum([exists(p) for p in pp]) == len(pp)

        if not feats_ready:

            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
            MMB = pd.read_csv('%s/members.csv' % self.data_dir)
            TRN = pd.read_csv('%s/train.csv' % self.data_dir)
            TST = pd.read_csv('%s/test.csv' % self.data_dir)

            self.logger.info('Merge SNG and SEI.')
            SNG = SNG.merge(SEI, on='song_id', how='left')

            self.logger.info('Merge TRN and TST with SNG and MMB.')
            TRN = TRN.merge(MMB, on='msno', how='left')
            TST = TST.merge(MMB, on='msno', how='left')
            TRN = TRN.merge(SNG, on='song_id', how='left')
            TST = TST.merge(SNG, on='song_id', how='left')

            self.logger.info('Combining TRN and TST')
            CMB = TRN.append(TST)

            # Throw away unused after merging.
            del SEI, MMB

            self.logger.info('Encoding user features')
            ukeys, ukeys2feats = get_user_feats(CMB)

            self.logger.info('Encoding song features')
            skeys, skeys2feats = get_song_feats(CMB)

            self.logger.info('Saving features')
            with open(path_feats, 'wb') as fp:
                feats = ukeys2feats.copy()
                feats.update(skeys2feats)
                pickle.dump(feats, fp)

            self.logger.info('Saving training keys')
            keys_trn = pd.DataFrame({
                'user': ukeys[:min(NTRN, len(TRN))],
                'song': skeys[:min(NTRN, len(TRN))],
                'target': TRN['target']
            })
            keys_trn.to_csv(path_sample_keys_trn, index=False)

            self.logger.info('Saving testing keys')
            keys_tst = pd.DataFrame({
                'id': TST['id'],
                'user': ukeys[-min(NTST, len(TST)):],
                'song': skeys[-min(NTST, len(TST)):]
            })
            keys_tst.to_csv(path_sample_keys_tst, index=False)

        # Read from disk and return keys and features.
        self.logger.info('Reading features from disk')
        keys = pd.read_csv(path_sample_keys_trn) if train \
            else pd.read_csv(path_sample_keys_tst)
        with open(path_feats, 'rb') as fp:
            feats = pickle.load(fp)

        return keys, feats

    def fit(self):

        # Get keys and features.
        samples, feats = self.get_features(train=True)

        # Generator to yield products of user features and song features.
        # In the current configuration there are ~7M * 7 * 9 interactions.
        # [user feature, song feature, target]
        # [['u-msno', 'hMLKXXl'], ['s-id', 'x+wwBW'], 0]
        # [['u-msno', 'hMLKXXl'], ['s-len', 2], 0]
        # [['u-msno', 'hMLKXXl'], ['s-lang', 3], 0]
        def product_gen(batch_size):
            user_keys = samples['user'].values
            song_keys = samples['song'].values
            targets = samples['target'].values
            while True:
                ii = np.random.permutation(len(samples))
                for ii_ in np.array_split(ii, ceil(len(samples) / batch_size)):
                    X, y = [], []
                    for i in ii_:
                        for p in product(feats[user_keys[i]], feats[song_keys[i]]):
                            X.append(p)
                            y.append(float(targets[i]))
                    yield X, y

        # Compute all of the unique feature keys.
        feats_unique = set(flatten(feats.values()))
        self.logger.info('Identified %d unique features' % len(feats_unique))

        # Instantiate the vector training model.
        model = GenericVectorSpace(feats_unique, vec_size=50)
        model.cuda()

        # Fit the vector space.
        batch_gen = product_gen(10000)
        # stop_func = lambda epoch, loss, auc: epoch > 10 or auc > 0.85
        # model.fit_generator(batch_gen, 10000, len(samples), stop_func)

        for i in range(10000):
            t0 = time()
            loss, auc = model.fit_batch(*next(batch_gen))
            print(i, round(loss, 5), round(auc, 5), round(time() - t0, 8))

        pdb.set_trace()

        # Pre-train user and song vectors to minimize:
        # error(sigmoid(dot(user feature vector, song feature vector)), target)

        # Train weighted model on pre-trained feature vectors. Either:
        # 1. interaction-constrained factorization machine.
        # 2. vector aggregator.

    pass


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--hpo', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = MultiVecRec(
        data_dir='data',
        artifacts_dir='artifacts/multivecrec',
        best_model_path='artifacts/multivecrec/model.hdf5',
        predict_path='artifacts/multivecrec/predict_tst_%d.csv' % int(time())
    )

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()

    if args['hpo']:
        model.hpo()
