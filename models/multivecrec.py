from collections import Counter
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

ICFM_HYPERPARAMS_DEFAULT = {
    'optimizer': SparseAdam,
    'optimizer_kwargs': {'lr': 0.01},
    'vec_size': 50,
    'vec_init_func': np.random.normal,
    'vec_init_kwargs': {'loc': 0, 'scale': 0.01},
}


class ICFM(nn.Module):

    def __init__(self, grouped_feat_names, feats_unique,
                 optimizer, optimizer_kwargs, vec_size,
                 vec_init_func, vec_init_kwargs, ):
        super(ICFM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Compute the possible feature interactions.
        # Feature interactions are denoted as tuples of feature names.
        self.inters = grouped_feat_names[0]
        for feat_names in grouped_feat_names[1:]:
            self.inters = list(product(self.inters, feat_names))
        self.inters = sorted(self.inters)
        self.logger.info('%d feature interactions' % len(self.inters))

        # Lookup feature interaction -> weight index.
        self.inter2idx = {f: i for i, f in enumerate(self.inters)}

        # Lookup feature -> vector index.
        self.feat2idx = {f: i for i, f in enumerate(feats_unique)}

        # Common vector space for all features.
        self.vecs = nn.Embedding(len(feats_unique), vec_size, sparse=True)

        # Initialize vector space  w/ given distribution and parameters.
        vec_init_kwargs.update({'size': (len(feats_unique), vec_size)})
        vw = vec_init_func(**vec_init_kwargs)
        self.vecs.weight.data = torch.FloatTensor(vw.astype(np.float32))

        # Initialize weights for each interaction.
        # TODO: make this initialization configurable.
        self.W_inter = Variable(torch.randn(len(self.inters)), requires_grad=True)

        # Initialize bias for all interactions.
        self.b_inter = Variable(torch.ones(1), requires_grad=True)

        # Training criteria.
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sample_batch):

        # feats_batch approximate shape (batch, no. groups, no. feats / group)
        # Example batch with one sample.
        # [[
        #     [
        #         ('u-msno', 'FGtll...'),
        #         ('u-age', 'MISSING'),
        #         ('u-city', 1),
        #         ('u-sex', 'MISSING'),
        #         ('u-reg-via', 7),
        #         ('u-reg-year', 2012),
        #         ('u-act-age', 5)],
        #     [
        #         ('s-id', 'BBzum...'),
        #         ('s-len', 1),
        #         ('s-lang', 52),
        #         ('s-year', 15),
        #         ('s-country', 'GB'),
        #         ('s-genre', 359),
        #         ('s-musician', 'Bastille'),
        #         ('s-musician', 'Dan Smith'),
        #         ('s-musician', 'Mark Crew')
        #     ]
        # ], ...]

        # 0  ('u-act-age', 's-country')
        # 1  ('u-act-age', 's-genre')
        # 2  ('u-act-age', 's-id')
        # 3  ('u-act-age', 's-lang')
        # 4  ('u-act-age', 's-len')
        # 5  ('u-act-age', 's-musician')
        # 6  ('u-act-age', 's-year')
        # 7  ('u-age', 's-country')
        # 8  ('u-age', 's-genre')
        # 9  ('u-age', 's-id')
        # 10 ('u-age', 's-lang')
        # 11 ('u-age', 's-len')
        # 12 ('u-age', 's-musician')
        # 13 ('u-age', 's-year')
        # 14 ('u-city', 's-country')
        # 15 ('u-city', 's-genre')
        # 16 ('u-city', 's-id')
        # 17 ('u-city', 's-lang')
        # 18 ('u-city', 's-len')
        # 19 ('u-city', 's-musician')
        # 20 ('u-city', 's-year')
        # 21 ('u-msno', 's-country')
        # 22 ('u-msno', 's-genre')
        # 23 ('u-msno', 's-id')
        # 24 ('u-msno', 's-lang')
        # 25 ('u-msno', 's-len')
        # 26 ('u-msno', 's-musician')
        # 27 ('u-msno', 's-year')
        # 28 ('u-reg-via', 's-country')
        # 29 ('u-reg-via', 's-genre')
        # 30 ('u-reg-via', 's-id')
        # 31 ('u-reg-via', 's-lang')
        # 32 ('u-reg-via', 's-len')
        # 33 ('u-reg-via', 's-musician')
        # 34 ('u-reg-via', 's-year')
        # 35 ('u-reg-year', 's-country')
        # 36 ('u-reg-year', 's-genre')
        # 37 ('u-reg-year', 's-id')
        # 38 ('u-reg-year', 's-lang')
        # 39 ('u-reg-year', 's-len')
        # 40 ('u-reg-year', 's-musician')
        # 41 ('u-reg-year', 's-year')
        # 42 ('u-sex', 's-country')
        # 43 ('u-sex', 's-genre')
        # 44 ('u-sex', 's-id')
        # 45 ('u-sex', 's-lang')
        # 46 ('u-sex', 's-len')
        # 47 ('u-sex', 's-musician')
        # 48 ('u-sex', 's-year')

        # # Determine the number of interactions.
        # nb_inters_max = max([len(g0) * len(g1) for g0, g1 in sample_batch])
        # sample_matrix = np.zeros((len(sample_batch), 4, nb_inters_max), dtype=np.int32)

        # for i, (g0, g1) in tqdm(enumerate(sample_batch)):

        #     inter_cntr = Counter()
        #     prod = product(g0, g1)

        #     for j, (f0, f1) in enumerate(prod):
        #         sample_matrix[i][0][j] = self.inter2idx[(f0[0], f1[0])]
        #         sample_matrix[i][1][j] = self.feat2idx[f0]
        #         sample_matrix[i][2][j] = self.feat2idx[f1]
        #         inter_cntr[sample_matrix[i][0][j]] += 1

        #     sample_matrix[i][3][:j] = [inter_cntr[ix] for ix in sample_matrix[i][0][:j]]

        pdb.set_trace()

        pass


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

    def fit(self, samples, feats, ii_trn=None, ii_val=None, ICFM_kwargs=ICFM_HYPERPARAMS_DEFAULT):

        # Compute all of the unique feature keys.
        feats_unique = set(flatten(feats.values()))
        self.logger.info('Identified %d unique features' % len(feats_unique))

        # Names of user features.
        feat_names_user, feat_names_song = set(), set()
        for feat_name, feat in feats_unique:
            if feat_name.startswith('u-'):
                feat_names_user.add(feat_name)
            elif feat_name.startswith('s-'):
                feat_names_song.add(feat_name)

        self.logger.info('User feature names: %s' % str(feat_names_user))
        self.logger.info('Song feature names: %s' % str(feat_names_song))

        icfm = ICFM(
            grouped_feat_names=[feat_names_user, feat_names_song],
            feats_unique=feats_unique,
            **ICFM_kwargs
        )

        user_keys = samples['user'].values
        song_keys = samples['song'].values
        targets = samples['target'].values

        X = [[feats[user_keys[i]], feats[song_keys[i]]] for i in range(10000)]
        icfm.forward(X)

        pdb.set_trace()

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
        model.fit(*model.get_features(train=True))

    if args['predict']:
        model.predict()

    if args['hpo']:
        model.hpo()
