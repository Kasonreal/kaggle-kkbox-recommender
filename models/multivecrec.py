from collections import Counter
from itertools import product
from math import log, ceil
from more_itertools import flatten
from multiprocessing import Pool, cpu_count
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


IAFM_HYPERPARAMS_DEFAULT = {
    'optimizer': optim.Adam,
    'optimizer_kwargs': {'lr': 0.01},
    'vec_size': 25,
    'vec_init_func': np.random.normal,
    'vec_init_kwargs': {'loc': 0, 'scale': 0.01},
}


def iafm_preprocess_batch_parallel(mpargs):

    keys, key2feats, intr2idx, feat2idx = mpargs

    intr_idxs = []  # Index for every interaction in the batch.
    intr_divs = []  # Divisor for each interaction weight for duplicate interactions.
    feat_idxs = []  # Feature indexes for pairs of features in the batch.
    smpl_idxs = []  # Indexes into above lists for every sample.

    # TODO: Many of the feat_idxs are duplicates and could be pruned.
    for k0, k1 in keys:
        smpl_idxs.append([])
        intr_cntr = Counter()
        for f0, f1 in product(key2feats[k0], key2feats[k1]):
            smpl_idxs[-1].append(len(intr_idxs))
            intr_idxs.append(intr2idx[(f0[0], f1[0])])
            feat_idxs.append([feat2idx[f0], feat2idx[f1]])
            intr_cntr[intr_idxs[-1]] += 1
        ntail = len(key2feats[k0] * len(key2feats[k1]))
        intr_divs += [intr_cntr[i] for i in intr_idxs[-ntail:]]

    return intr_idxs, intr_divs, feat_idxs, smpl_idxs


class IAFM(nn.Module):

    def __init__(self, grouped_feat_names, feats_unique,
                 optimizer, optimizer_kwargs, vec_size,
                 vec_init_func, vec_init_kwargs, ):
        super(IAFM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Compute the possible feature interactions.
        # Feature interactions are denoted as tuples of feature names.
        self.intrs = grouped_feat_names[0]
        for feat_names in grouped_feat_names[1:]:
            self.intrs = list(product(self.intrs, feat_names))
        self.logger.info('%d feature interactions' % len(self.intrs))

        # Lookup feature interaction -> weight index.
        self.intr2idx = {f: i for i, f in enumerate(self.intrs)}

        # Lookup feature -> vector index.
        self.feat2idx = {f: i for i, f in enumerate(feats_unique)}

        # Common vector space for all features.
        self.vecs = nn.Embedding(len(feats_unique), vec_size, sparse=False)

        # Initialize vector space  w/ given distribution and parameters.
        vec_init_kwargs.update({'size': (len(feats_unique), vec_size)})
        vw = vec_init_func(**vec_init_kwargs)
        self.vecs.weight.data = torch.FloatTensor(vw.astype(np.float32))

        # Initialize weights for each interaction.
        # TODO: make this initialization configurable.
        # TODO: make this use nn.Parameter instead of Embedding.
        self.intr_W = nn.Embedding(len(self.intrs), 1, sparse=False)
        self.intr_W.weight.data.normal_(0, 1.0)

        # Initialize bias for all interactions.
        # TODO: make this initialization configurable.
        self.intr_b = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Training criteria.
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def preprocess_batch(self, keys, key2feats):
        '''Convert the keys batch into a format that can be efficiently computed.'''

        x = iafm_preprocess_batch_parallel((keys, key2feats, self.intr2idx, self.feat2idx))
        intr_idxs, intr_divs, feat_idxs, smpl_idxs = x

        # Convert indexes to torch-friendly types.
        intr_idxs_ch = Variable(torch.LongTensor(intr_idxs)).cuda()
        intr_divs_ch = Variable(torch.FloatTensor(intr_divs)).cuda()
        feat_idxs_ch = Variable(torch.LongTensor(feat_idxs)).cuda()
        smpl_idxs_ch = [Variable(torch.LongTensor(x)).cuda() for x in smpl_idxs]

        return intr_idxs_ch, intr_divs_ch, feat_idxs_ch, smpl_idxs_ch

    def forward_batch(self, intr_idxs_ch, intr_divs_ch, feat_idxs_ch, smpl_idxs_ch):

        # Use the feat_idxs to retrieve vectors.
        v = self.vecs(feat_idxs_ch)

        # Dot products across sample axis.
        d = torch.sum(torch.prod(v, dim=1), dim=1)

        # Use the intr_idxs to retrieve interaction weights.
        w = self.intr_W(intr_idxs_ch)

        # Adjust each interaction weight by the interaction frequency for that sample.
        # Multiply each dot product by its adjusted interaction weight.
        # Add global bias term.
        wdb = w[:, 0] / intr_divs_ch * d + self.intr_b

        # Sum each sample's indexes to get a scalar for each sample.
        outputs = Variable(torch.zeros(len(smpl_idxs_ch)))
        for i, idxs_ch in enumerate(smpl_idxs_ch):
            outputs[i] = torch.sum(wdb[idxs_ch].cpu())
        return outputs

    def fit_batch(self, keys, yt, feats):

        # TODO: Update vectors for missing values.

        # Convert keys and feats into torch variables for the forward pass.
        t0 = time()
        batch_ch = self.preprocess_batch(keys, feats)
        print(time() - t0)

        # Forward pass.
        t0 = time()
        self.optimizer.zero_grad()
        yt_ch = Variable(torch.FloatTensor(yt * 1.))
        yp_ch = self.forward_batch(*batch_ch)
        print(time() - t0)

        # Loss, backprop, update.
        t0 = time()
        loss = self.criterion(yp_ch, yt_ch)
        loss.backward()
        self.optimizer.step()
        print(time() - t0)

        # Metrics.
        t0 = time()
        loss = loss.cpu().data.numpy()[0]
        yp = self._sigmoid(yp_ch.data.numpy())
        auc = roc_auc_score(yt, yp)
        print(time() - t0)

        return loss, auc

    def predict_batch(self, keys, feats, yt=None):
        batch_ch = self.preprocess_batch(keys, feats)
        yp_ch = self.forward_batch(*batch_ch)
        yp = self._sigmoid(yp_ch.data.numpy())

        if yt is not None:
            auc = roc_auc_score(yt, yp)
            return yp, auc

        return yp


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

    def fit(self, samples, feats, ii_trn=None, ii_val=None, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):

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

        iafm = IAFM(
            grouped_feat_names=[feat_names_user, feat_names_song],
            feats_unique=feats_unique,
            **IAFM_kwargs
        )
        iafm.cuda()

        keys = samples[['user', 'song']].values
        targets = samples['target'].values

        p = np.random.permutation(len(keys))
        ii_trn = p[:1000000]
        ii_val = p[1000000:1100000]

        for e in range(5):
            np.random.shuffle(ii_trn)
            ii_batches = np.array_split(ii_trn, ceil(len(ii_trn) / 10000))
            for i, ii_ in enumerate(ii_batches):
                t0 = time()
                loss, acc = iafm.fit_batch(keys[ii_], targets[ii_], feats)
                print('%-3d %-5d %.4lf %.4lf %.4lf' % (e, i, loss, acc, time() - t0))
                print('*' * 10)

            ii_batches = np.array_split(ii_val, ceil(len(ii_val) / 10000))
            mean_auc = 0.
            for i, ii_ in enumerate(ii_batches):
                yp, auc = iafm.predict_batch(keys[ii_], feats, targets[ii_])
                mean_auc += auc / len(ii_batches)

            print('%-3d val auc=%.4lf' % (e, mean_auc))

        pdb.set_trace()


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
