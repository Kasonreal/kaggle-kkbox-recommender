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
MISSING = 0


def round_to(n, r):
    return round(n / r) * r


def encode(series, is_missing=lambda x: False, split_multi=lambda x: [x], missing_label=MISSING):
    # vals = ['a', 'b', 'c']
    # print(vals, encode(vals))
    # vals = ['a|b', 'b', 'c', 'c|d']
    # print(vals, encode(vals, split_multi=lambda x: x.split('|')))
    # vals = ['a', 'b', 'a|b', 'no']
    # print(vals, encode(vals, split_multi=lambda x: x.split('|'), is_missing=lambda x: x == 'no'))

    raw_to_label = dict()
    next_label = missing_label + 1
    labeled = []

    for raw in series:

        # Prepare empty list for this value.
        labeled.append([])

        # If the value meets "missing" criteria, just set the missing label.
        if is_missing(raw):
            labeled[-1].append(missing_label)
            continue

        # Split the raw value into potentially several values.
        for raw_ in split_multi(raw):

            # Otherwise try to get its label.
            label = raw_to_label.get(raw_)

            # If it's not defined, define it.
            if label is None:
                label = next_label
                raw_to_label[raw_] = next_label
                next_label += 1

            # Set the label.
            labeled[-1].append(label)

    return labeled


def df_to_user_feats(df):
    """ Returning:
    ids: sequential ids for each unique user.
    feats: mapping user id -> [(feature type, feature value), ...]
    names: names of the feature types.
    """

    # Remove duplicate users.
    df = df.drop_duplicates('msno')

    # Ids are used to lookup the users in the feature map.
    ids = list(range(len(df)))

    # Each user's id maps to a list of features.
    feats = {id_: [] for id_ in ids}
    names = []  # Names of the features.

    # User index.
    for i, f in zip(ids, encode(df['msno'])):
        feats[i].append((len(names), *f))
    names.append('user-index')

    # User ages. Round to multiples of 5.
    # Values clipped outside a reasonable range.
    is_missing = lambda x: x <= 5 or x > 70
    for i, f in zip(ids, encode(round_to(df['bd'], 5), is_missing)):
        feats[i].append((len(names), *f))
    names.append('user-age')

    # User city. No missing values.
    for i, f in zip(ids, encode(df['city'])):
        feats[i].append((len(names), *f))
    names.append('user-city')

    # User gender. Missing if not 'female' or 'male'.
    is_missing = lambda x: x not in {'female', 'male'}
    for i, f in zip(ids, encode(df['gender'], is_missing)):
        feats[i].append((len(names), *f))
    names.append('user-gender')

    # User registration method. No missing values.
    for i, f in zip(ids, encode(df['registered_via'])):
        feats[i].append((len(names), *f))
    names.append('user-registration-method')

    # User account age, in years. No missing values.
    y0 = df['registration_init_time'].apply(lambda x: int(str(x)[: 4])).values
    y1 = df['expiration_date'].apply(lambda x: int(str(x)[: 4])).values
    for i, f in zip(ids, encode(y1 - y0)):
        feats[i].append((len(names), *f))
    names.append('user-account-age')

    # User registration year. No missing values.
    for i, f in zip(ids, encode(y0)):
        feats[i].append((len(names), *f))
    names.append('user-registration-year')

    return ids, feats, names


def df_to_item_feats(df):

    names, feats = [], []

    # Item id.
    t0 = time()
    feats.append(encode(df['song_id']))
    names.append('item-id')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item length. Missing if nan. Convert to minutes, log scale, round.
    t0 = time()
    x = df['song_length'] / 1000 / 60
    x = np.log(np.clip(x, 0, x.max()) + 1).round()
    feats.append(encode(x, is_missing=np.isnan))
    names.append('item-length')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item language. Missing if nan.
    t0 = time()
    feats.append(encode(df['language'], is_missing=np.isnan))
    names.append('item-language')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item year. Missing if nan. Split into 3-year intervals.
    t0 = time()
    x = df['isrc'].apply(lambda x: int(x[5: 7]) if type(x) == str else x)
    x = round_to(x, 3)
    feats.append(encode(x, is_missing=np.isnan))
    names.append('item-year')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item country.
    t0 = time()
    x = df['isrc'].apply(lambda x: x[: 2] if type(x) == str else x)
    feats.append(encode(x, is_missing=lambda x: type(x) is not str))
    names.append('item-country')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item genre. Missing if nan. Split on pipes.
    t0 = time()
    is_missing = lambda x: type(x) is not str
    split_multi = lambda x: [s.strip() for s in x.split('|')]
    feats.append(encode(df['genre_ids'], is_missing, split_multi))
    names.append('item-genres')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    # Item musicians. Combine artist, composer, lyricist into a single feature.
    # Missing if nan. Split on pipes.
    x = df.apply(lambda row: '|'.join([
        row['artist_name'] if type(row['artist_name']) == str else '',
        row['lyricist'] if type(row['lyricist']) == str else '',
        row['composer'] if type(row['composer']) == str else '',
    ]), axis=1)
    feats.append(encode(x, is_missing, split_multi))
    names.append('item-musician')
    print('%-30s %.3lf sec.' % (names[-1], time() - t0))

    return names, feats_cols_to_rows(feats)


def df_to_ctxt_feats(df):
    """ Returning:
    ids: sequential ids for each context.
    feats: mapping context id -> [(feature type, feature value), ...]
    names: names of the feature types.
    """

    # Context is defined as a combination of three columns.
    # Remove the duplicate contexts.
    df = df.drop_duplicates(['source_screen_name', 'source_system_tab', 'source_type'])

    # Ids are used to lookup the users in the feature map.
    ids = list(range(len(df)))

    # Each context's id maps to a list of features.
    feats = {id_: [] for id_ in ids}
    names = []

    # Context screen name. Missing if "Unknown" or nan.
    is_missing = lambda x: x == 'Unknown' or type(x) is not str
    for i, f in zip(ids, encode(df['source_screen_name'], is_missing)):
        feats[i].append((len(names), *f))
    names.append('ctxt-screen-name')

    # Context system tab. Missing if "null" or nan.
    is_missing = lambda x: x == 'null' or type(x) is not str
    for i, f in zip(ids, encode(df['source_system_tab'], is_missing)):
        feats[i].append((len(names), *f))
    names.append('ctxt-system-tab')

    # Context source type. Missing if nan.
    is_missing = lambda x: type(x) is not str
    for i, f in zip(ids, encode(df['source_type'], is_missing)):
        feats[i].append((len(names), *f))
    names.append('ctxt-source-type')

    return ids, feats, names


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

        path_ids_trn = '%s/data-ids-trn.pkl' % self.artifacts_dir
        path_ids_tst = '%s/data-ids-tst.pkl' % self.artifacts_dir
        path_feats_user = '%s/data-feats-user.pkl' % self.artifacts_dir
        path_feats_item = '%s/data-feats-item.pkl' % self.artifacts_dir
        path_feats_ctxt = '%s/data-feats-ctxt.pkl' % self.artifacts_dir

        feats_on_disk = exists(path_feats_user)
        feats_on_disk = feats_on_disk and exists(path_feats_item)
        feats_on_disk = feats_on_disk and exists(path_feats_ctxt)

        if not feats_on_disk:

            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
            MMB = pd.read_csv('%s/members.csv' % self.data_dir)
            TRN = pd.read_csv('%s/train.csv' % self.data_dir, nrows=100000)
            TST = pd.read_csv('%s/test.csv' % self.data_dir, nrows=100000)

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
            del SEI, MMB, TRN, TST

            CMB.drop('id', inplace=True, axis=1)
            CMB.drop('target', inplace=True, axis=1)

            t0 = time()
            self.logger.info('Encoding user features (%d sec)' % (time() - t0))
            ids, feats, names = df_to_user_feats(CMB)
            with open(path_feats_user, 'wb') as fp:
                pickle.dump((names, feats), fp)
            self.logger.info('%d seconds' % (time() - t0))

            # self.logger.info('Encoding item features')
            # names, feats = df_to_item_feats(CMB)
            # with open(path_feats_item, 'wb') as fp:
            #     pickle.dump((names, feats), fp)

            t0 = time()
            self.logger.info('Encoding context features')
            ids, feats, names = df_to_ctxt_feats(CMB)
            with open(path_feats_ctxt, 'wb') as fp:
                pickle.dump((feats, names), fp)
            self.logger.info('%d seconds' % (time() - t0))

        def load_pickle(path):
            with open(path, 'rb') as fp:
                return pickle.load(fp)

        if train:
            usn, usf = load_pickle(path_feats_user)
            pdb.set_trace()

            itn, itf = load_pickle(path_feats_item)
            ctn, ctf = load_pickle(path_feats_ctxt)
            return usn, usf[:NTRN], itn, itf[:NTRN], ctn, ctf[:NTRN]

        if test:
            usn, usf = load_pickle(path_feats_user)
            itn, itf = load_pickle(path_feats_item)
            ctn, ctf = load_pickle(path_feats_ctxt)
            return usn, usf[-NTST:], itn, itf[-NTST:], ctn, ctf[-NTST:]

    def fit(self):

        feats = self.get_features()
        user_names, user_feats, item_names, item_feats, ctxt_names, ctxt_feats = feats

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

    F = model.get_features(train=True)
    pdb.set_trace()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()

    if args['hpo']:
        model.hpo()
