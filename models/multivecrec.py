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

assert getenv('CUDA_VISIBLE_DEVICES') is not None, "Specify a GPU"
assert len(getenv('CUDA_VISIBLE_DEVICES')) > 0, "Specify a GPU"

IAFM_HYPERPARAMS_DEFAULT = {

    'optimizer': optim.Adam,
    # Must use regular (not sparse) Adam. See bug:
    # https://discuss.pytorch.org/t/bug-of-nn-embedding-when-sparse-true-and-padding-idx-is-set/9382/2

    'optimizer_kwargs': {'lr': 0.01},
    'vec_size': 60,
    'vec_init_func': np.random.normal,
    'vec_init_kwargs': {'loc': 0, 'scale': 0.01},
    'nb_epochs_max': 100,
    'batch_size': 40000,
    'early_stop_delta': 0.05,
    'early_stop_patience': 2,
}


class IAFM(nn.Module):

    def __init__(self, key2feats, optimizer, optimizer_kwargs, vec_size,
                 vec_init_func, vec_init_kwargs, nb_epochs_max, batch_size,
                 early_stop_delta, early_stop_patience):
        super(IAFM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Keep the key -> features mapping.
        self.key2feats = key2feats

        # Compute all of the unique feature keys.
        feats_unique = set(flatten(self.key2feats.values()))
        self.logger.info('Identified %d unique features' % len(feats_unique))

        # Map feature -> vector index. Add 1 to account for padding index.
        self.feat2ii = {f: i + 1 for i, f in enumerate(feats_unique)}

        # One vector space for all features. One vector is kept empty for padding.
        nb_vecs = max(self.feat2ii.values()) + 1
        self.vecs = nn.Embedding(nb_vecs, vec_size, padding_idx=0, sparse=False).cuda()
        self.PADDING_VEC_I = self.vecs.padding_idx

        # Initialize vector space  w/ given distribution and parameters.
        vec_init_kwargs.update({'size': tuple(self.vecs.weight.size())})
        vw = vec_init_func(**vec_init_kwargs)
        self.vecs.weight.data = torch.FloatTensor(vw.astype(np.float32))

        # Training criteria.
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

        # Training details.
        self.nb_epochs_max = nb_epochs_max
        self.batch_size = batch_size
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience

    @staticmethod
    def _sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def fit(self, key_pairs_trn, targets_trn, key_pairs_val=[], targets_val=None):

        # Map key -> [padded list of the indexes].
        # Map key -> number of valid (non-padded) indexes in index list.
        nb_ii_max = max(map(len, self.key2feats.values()))
        key2vii, key2len = {}, {}
        for k, v in self.key2feats.items():
            padding = [self.PADDING_VEC_I] * (nb_ii_max - len(v))
            key2vii[k] = [self.feat2ii[f] for f in v] + padding
            key2len[k] = len(v)

        # Now all lists of indexes have equivalent length.
        assert len(set(map(len, key2vii.values()))) == 1

        nb_batches_trn = ceil(len(key_pairs_trn) / self.batch_size)
        nb_batches_val = ceil(len(key_pairs_val or []) / self.batch_size)

        for ei in range(self.nb_epochs_max):

            # Batched permutation of indexes.
            ii_trn = np.random.permutation(len(key_pairs_trn))
            ii_trn = np.array_split(ii_trn, nb_batches_trn)

            # Iterate through permutation of training samples.
            for bi, ii_trn_ in enumerate(ii_trn):
                t0 = time()
                pt, loss = self.forward(key_pairs_trn[ii_trn_, 0],
                                        key_pairs_trn[ii_trn_, 1],
                                        key2vii, key2len,
                                        targets_trn[ii_trn_])
                auc = roc_auc_score(targets_trn[ii_trn_], pt)
                print('%-3d %-3d %.3lf %.3lf %.4lf' %
                      (ei, bi, loss, auc, time() - t0))

            pass

    def forward(self, kk0, kk1, key2vii, key2len, tt=None):

        ii0, ii1, nb_inters = [], [], []
        for k0, k1 in zip(kk0, kk1):
            ii0.append(key2vii[k0])
            ii1.append(key2vii[k1])
            nb_inters.append(key2len[k0] * key2len[k1])

        # Convert to torch variables.
        ii0_ch = Variable(torch.LongTensor(ii0)).cuda()
        ii1_ch = Variable(torch.LongTensor(ii1)).cuda()

        # Retrieve as two batches of vectors.
        vv0_ch = self.vecs(ii0_ch)
        vv1_ch = self.vecs(ii1_ch)

        # Matrix multiply to get all possible vector interaction dot products.
        # The padding vectors' products will be 0.
        muls_ch = torch.matmul(vv0_ch, vv1_ch.transpose(2, 1))

        # Sum each sample's interactions.
        sums_ch = muls_ch.sum(-1).sum(-1)

        # Divide by the number of interactions in each sample to get predicted target.
        nb_inters_ch = Variable(torch.FloatTensor(nb_inters), requires_grad=False).cuda()
        pt_ch = sums_ch / nb_inters_ch
        pt = self._sigmoid(pt_ch.cpu().data.numpy())

        # No true targets given, just return predicted targets.
        if tt is None:
            return pt

        # Compute the loss and make a gradient update.
        tt_ch = Variable(torch.FloatTensor(tt * 1.)).cuda()
        loss_ch = self.criterion(pt_ch, tt_ch)
        loss_ch.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss_ch.cpu().data.numpy()[0]
        return pt, loss


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
    key2feats = {}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # User msno (id).
        key2feats[k].append(('u-msno', row['msno']))

        # User age. Clipped and rounded.
        if 0 < row['bd'] < 70:
            key2feats[k].append(('u-age', round_to(row['bd'], 5)))

        # User city. No missing values.
        key2feats[k].append(('u-city', int(row['city'])))

        # User gender. Missing if not female or male.
        if row['gender'] in {'male', 'female'}:
            key2feats[k].append(('u-sex', row['gender']))

        # User registration method. No missing values.
        key2feats[k].append(('u-reg-via', int(row['registered_via'])))

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        key2feats[k].append(('u-reg-year', y0))

    return keys, key2feats


def split_multi(maybe_vals):
    if type(maybe_vals) == str:
        split = maybe_vals.split('|')
        try:
            return [int(x) for x in split]
        except ValueError as ex:
            return [x.strip() for x in split]
    return []


def get_song_feats(df):

    # Key is the song id prefixed by "s-"
    songid2key = lambda x: 's-%s' % x
    keys = df['song_id'].apply(songid2key).values.tolist()

    # Count the instances of each musician and genre.
    # Musicians encompass artist, composer, and lyricist.
    t0 = time()
    pool = Pool(cpu_count())
    musicians = flatten(pool.map(split_multi,
                                 df['artist_name'].values.tolist() +
                                 df['lyricist'].values.tolist() +
                                 df['composer'].values.tolist()))
    musician_counts = Counter(musicians)
    print('%.4lf' % (time() - t0))

    genre_ids = flatten(pool.map(split_multi, df['genre_ids'].values.tolist()))
    genre_id_counts = Counter(genre_ids)
    print('%.4lf' % (time() - t0))
    pool.close()

    # Remove duplicates.
    df = df.drop_duplicates('song_id')
    keys_dedup = df['song_id'].apply(songid2key).values.tolist()

    # Build mapping from unique keys to features.
    key2feats = {}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # Song id.
        key2feats[k].append(('s-id', row['song_id']))

        # Song length. Missing if nan.
        if not np.isnan(row['song_length']):
            f = row['song_length'] / 1000 / 60
            f = round(log(max(1, f)))
            key2feats[k].append(('s-len', f))

        # Song language. Missing if nan.
        if not np.isnan(row['language']):
            key2feats[k].append(('s-lang', int(row['language'])))

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is str:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            key2feats[k].append(('s-year', f))
            key2feats[k].append(('s-country', row['isrc'][:2]))

        # Song genre(s). Missing if nan. Split on pipes.
        gg = split_multi(row['genre_ids'])
        if len(gg) > 0:
            ggc = map(genre_id_counts.get, gg)
            key2feats[k].append(('s-genre', gg[np.argmax(ggc)]))

        # TODO: consider artist, lyricist, musician separately.
        mm = split_multi(row['artist_name'])
        mm += split_multi(row['lyricist'])
        mm += split_multi(row['composer'])
        if len(mm) > 0:
            mmc = map(musician_counts.get, mm)
            key2feats[k].append(('s-musician', mm[np.argmax(mmc)]))

    return keys, key2feats


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

            self.logger.info('Encoding song features')
            skeys, skey2feats = get_song_feats(CMB)

            self.logger.info('Encoding user features')
            ukeys, ukey2feats = get_user_feats(CMB)

            self.logger.info('Saving features')
            with open(path_feats, 'wb') as fp:
                feats = ukey2feats.copy()
                feats.update(skey2feats)
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

    def fit(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        iafm = IAFM(key2feats, **IAFM_kwargs)
        iafm.cuda()
        key_pairs = samples[['user', 'song']].values
        targets = samples['target'].values
        iafm.fit(key_pairs, targets)

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
