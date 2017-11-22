from itertools import product
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from multiprocessing import cpu_count, Pool
from more_itertools import flatten
from os.path import exists
from pprint import pformat
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
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

NTHR = cpu_count()
NTRN = 7377418
NTST = 2556790
np.random.seed(865)


def round_to(n, r):
    return round(n / r) * r


# def encode(series, offset=0):
#     lookup = dict()
#     encoded = np.empty((len(series), 0)).tolist()
#     for i, raw in enumerate(series):
#         encoding = lookup.get(raw)
#         if encoding is not None:
#             encoded[i].append(offset + encoding)
#         else:
#             encoded[i].append(offset + len(lookup))
#             lookup[raw] = len(lookup)
#     return encoded, offset + len(lookup)

def encode(series, offset=0):
    lookup = dict()
    encoded = [None] * len(series)
    for i, raw in enumerate(series):
        encoding = lookup.get(raw)
        if encoding is not None:
            encoded[i] = offset + encoding
        else:
            encoded[i] = offset + len(lookup)
            lookup[raw] = len(lookup)
    return encoded, offset + len(lookup)

# a, b = encode(['a', 'b', 'b', 'c', 'c', 'c', 'a', 'd'], offset=0)
# print(a, b)
# c, d = encode(['e', 'f', 'g', 'g'], offset=b)
# print(c, d)
# pdb.set_trace()


def df_to_user_feats(df):

    # User id.
    t0 = time()
    df['msno'], offset = encode(df['msno'], 0)
    print('msno', time() - t0)

    # User age.
    t0 = time()
    df['bd'] = np.clip(df['bd'], 0, 50)
    df['bd'] = round_to(df['bd'], 5)
    df['bd'], offset = encode(df['bd'], offset)
    print('bd', time() - t0)

    # User city.
    t0 = time()
    df['city'], offset = encode(df['city'], offset)
    print('city', time() - t0)

    # User gender.
    t0 = time()
    df['gender'].fillna('unk', inplace=True)
    df['gender'], offset = encode(df['gender'], offset)
    print('gender', time() - t0)

    # User registration method.
    t0 = time()
    df['registered_via'], offset = encode(df['registered_via'], offset)
    print('registered_via', time() - t0)

    # User account age, in years.
    t0 = time()
    y0 = df['registration_init_time'].apply(lambda x: int(str(x)[:4])).values
    y1 = df['expiration_date'].apply(lambda x: int(str(x)[:4])).values
    df['account_age'] = y1 - y0
    df['account_age'], offset = encode(df['account_age'], offset)
    print('account_age', time() - t0)

    # User registration year.
    t0 = time()
    df['registration_init_time'] = df['registration_init_time'].apply(lambda x: str(x)[:4])
    df['registration_init_time'], offset = encode(df['registration_init_time'], offset)
    print('registration_init_time', time() - t0)

    cc = ['msno', 'bd', 'city', 'gender', 'registered_via', 'account_age', 'registration_init_time']
    return df[cc]


def df_to_item_feats(df):

    # Item id.
    df['song_id'], offset = encode(df['song_id'], 0)

    # Item length. Converted to seconds and passed through log to get less variety.
    df['song_length'].fillna(df['song_length'].median(), inplace=True)
    df['song_length'] = np.log(df['song_length'] / 1000 + 1).round()
    df['song_length'], offset = encode(df['song_length'], offset)

    # Item language.
    imp = df['language'].value_counts().idxmax()
    df['language'].fillna(imp, inplace=True)
    df['language'], offset = encode(df['language'], offset)

    # Item year (via ISRC) split into 3-year intervals.
    isrc_ = df['isrc'].dropna()
    df['year'] = isrc_.apply(lambda x: round_to(int(x[5:7]), 3))
    df['year'], offset = encode(df['year'], offset)

    # Item country (via ISRC).
    df['country'] = isrc_.apply(lambda x: x[:2])
    df['country'], offset = encode(df['country'], offset)

    # Item genre. For items with multiple genres, pick the most frequent one.

    # i_gen*
    # i_art*
    # i_cmp*
    # i_lyr*

    pass


def df_to_ctxt_feats(df):

    imp = df['source_screen_name'].value_counts().idxmax()
    df['source_screen_name'].replace('Unknown', imp, inplace=True)
    df['source_screen_name'].fillna(imp, inplace=True)
    df['source_screen_name'], offset = encode(df['source_screen_name'])

    imp = df['source_system_tab'].value_counts().idxmax()
    df['source_system_tab'].replace('null', imp, inplace=True)
    df['source_system_tab'].fillna(imp, inplace=True)
    df['source_system_tab'], offset = encode(df['source_system_tab'], offset)

    imp = df['source_type'].value_counts().idxmax()
    df['source_type'].fillna(imp, inplace=True)
    df['source_type'], offset = encode(df['source_type'], offset)

    cc = ['source_screen_name', 'source_system_tab', 'source_type']
    return df[cc]


class UICVecRec(object):
    """Get SpecVecRec't"""

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

        path_user_feats = '%s/feats-user.pkl' % self.artifacts_dir
        path_item_feats = '%s/feats-item.pkl' % self.artifacts_dir
        path_ctxt_feats = '%s/feats-ctxt.pkl' % self.artifacts_dir

        if train\
                and exists(path_user_feats) \
                and exists(path_item_feats) \
                and exists(path_ctxt_feats):
            U = np.load(path_user_feats)[:NTRN]
            I = np.load(path_item_feats)[:NTRN]
            C = np.load(path_ctxt_feats)[:NTRN]
            return U, I, C

        if test\
                and exists(path_user_feats) \
                and exists(path_item_feats) \
                and exists(path_ctxt_feats):
            U = np.load(path_user_feats)[:NTST]
            I = np.load(path_item_feats)[:NTST]
            C = np.load(path_ctxt_feats)[:NTST]
            return U, I, C

        t0 = time()
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

        for c in ['genre_ids', 'artist_name', 'composer', 'lyricist']:
            CMB[c].fillna('unk', inplace=True)
            cnts = CMB[c].apply(lambda x: x.count('|') + 1)
            print(c, round(np.sum(cnts > 1) / len(CMB), 3), np.median(cnts), max(cnts), np.argmax(cnts))

        pdb.set_trace()

        # Throw away unused after merging.
        del SEI, MMB, TRN, TST

        CMB.drop('id', inplace=True, axis=1)
        CMB.drop('target', inplace=True, axis=1)

        # self.logger.info('Encoding user features')
        # U = df_to_user_feats(CMB)
        # U.to_pickle(path_user_feats)
        # self.logger.info('Saved %s (%d sec.)' % (path_user_feats, time() - t0))

        I = df_to_item_feats(CMB)
        I.to_pickle(path_item_feats)
        self.logger.info('Saved %s (%d sec.)' % (path_item_feats, time() - t0))

        # C = df_to_ctxt_feats(CMB)
        # C.to_pickle(path_ctxt_feats)
        # self.logger.info('Saved %s (%d sec.)' % (path_ctxt_feats, time() - t0))

        pdb.set_trace()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--hpo', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = UICVecRec(
        data_dir='data',
        artifacts_dir='artifacts/uicvecrec',
        best_model_path='artifacts/uicvecrec/model.hdf5',
        predict_path='artifacts/uicvecrec/predict_tst_%d.csv' % int(time())
    )

    model.get_features()

    # if args['fit']:
    #     model.fit()

    # if args['predict']:
    #     model.predict()

    # if args['hpo']:
    #     model.hpo()
