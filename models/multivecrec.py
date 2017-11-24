from collections import Counter
from math import log
from more_itertools import flatten
from os.path import exists
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
import torch.nn as nn
import torch.optim as optim

NTRN = 7377418
NTST = 2556790
np.random.seed(865)

# Reserved index for missing values.
MISSING = 'MISSING'


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
        keys2feats[k].append(['u-msno', row['msno']])

        # User age. Clipped and rounded.
        if not (0 < row['bd'] < 70):
            keys2feats[k].append(['u-age', MISSING])
        else:
            keys2feats[k].append(['u-age', round_to(row['bd'], 5)])

        # User city. No missing values.
        keys2feats[k].append(['u-city', int(row['city'])])

        # User gender. Missing if not female or male.
        if row['gender'] not in {'male', 'female'}:
            keys2feats[k].append(['u-sex', MISSING])
        else:
            keys2feats[k].append(['u-sex', row['gender']])

        # User registration method. No missing values.
        keys2feats[k].append(['u-reg-via', int(row['registered_via'])])

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        keys2feats[k].append(['u-reg-year', y0])

        # User account age, in years. No missing values.
        y1 = int(str(row['expiration_date'])[:4])
        keys2feats[k].append(['u-act-age', y1 - y0])

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
        keys2feats[k].append(['s-id', row['song_id']])

        # Song length. Missing if nan.
        if np.isnan(row['song_length']):
            keys2feats[k].append(['s-len', MISSING])
        else:
            f = row['song_length'] / 1000 / 60
            f = round(log(max(1, f)))
            keys2feats[k].append(['s-len', f])

        # Song language. Missing if nan.
        if np.isnan(row['language']):
            keys2feats[k].append(['s-lang', MISSING])
        else:
            keys2feats[k].append(['s-lang', int(row['language'])])

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is not str:
            keys2feats[k].append(['s-year', MISSING])
            keys2feats[k].append(['s-country', MISSING])
        else:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            keys2feats[k].append(['s-year', f])
            keys2feats[k].append(['s-country', row['isrc'][:2]])

        # Song genre(s). Missing if nan. Split on pipes.
        if type(row['genre_ids']) is not str:
            keys2feats[k].append(['s-genre', MISSING])
        else:
            for g in row['genre_ids'].split('|'):
                keys2feats[k].append(['s-genre', int(g)])

        # Song musicians. Combine artist, composer, lyricist.
        if str not in {type(row['artist_name']), type(row['composer']), type('lyricist')}:
            keys2feats[k].append('s-musician', MISSING)

        # Artist. Missing if nan, but already accounted for.
        if type(row['artist_name']) == str:
            for m in row['artist_name'].split('|'):
                keys2feats[k].append(['s-musician', m.strip()])

        # Composer. Missing if nan, but already accounted for.
        if type(row['composer']) == str:
            for m in row['composer'].split('|'):
                keys2feats[k].append(['s-musician', m.strip()])

        # Lyricist. Missing if nan, but already accounted for.
        if type(row['lyricist']) == str:
            for m in row['lyricist'].split('|'):
                keys2feats[k].append(['s-musician', m.strip()])

    return keys, keys2feats


class Net(nn.Module):

    def __init__(self):

        return

    def forward(self):

        return


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
        self.logger = logging.getLogger('VecRec')

    def get_features(self, train=False, test=False):

        path_sample_keys_trn = '%s/data-sample-keys-trn.csv' % self.artifacts_dir
        path_sample_keys_tst = '%s/data-sample-keys-tst.csv' % self.artifacts_dir
        path_feats = '%s/data-feats.json' % self.artifacts_dir

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
            with open(path_feats, 'w') as fp:
                feats = ukeys2feats.copy()
                feats.update(skeys2feats)
                json.dump(feats, fp, ensure_ascii=False, sort_keys=True, indent=2)

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
        keys = pd.read_csv(path_sample_keys_trn) if train else pd.read_csv(path_sample_keys_tst)
        with open(path_feats) as fp:
            feats = json.load(fp)

        return keys, feats

    def fit(self):

        # Get keys and features.

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

    F = model.get_features(train=True)
    pdb.set_trace()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()

    if args['hpo']:
        model.hpo()
