from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split
from multiprocessing import cpu_count
from more_itertools import flatten
from os import getenv
from os.path import exists
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pdb
import sys

np.random.seed(865)


class LFMRec(object):
    """Get SpecVecRec't"""

    def __init__(self,
                 data_dir,
                 artifacts_dir):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.logger = logging.getLogger('VecRec')

    def get_features(self, train=True, test=False):
        """
        """

        def round_to(n, r):
            return round(n / r) * r

        def song_to_tags(row):
            tags = []

            if type(row['artist_name']) == str:
                tags += ['artist:%s' % x for x in row['artist_name'].lower().split('|')]

            if type(row['genre_ids']) == str:
                tags += ['genre:%s' % x for x in row['genre_ids'].lower().split('|')]

            if type(row['isrc']) == str:
                tags.append('country:%s' % row['isrc'][:2])
                tags.append('year:%d' % round_to(int(row['isrc'][5:7]), 5))

            if not np.isnan(row['language']):
                tags.append('language:%d' % int(row['language']))

            return tags

        def user_to_tags(row, train=False, test=False):
            tags = []

            tags.append('age:%d' % round_to(np.clip(row['bd'], 0, 50), 5))

            if not np.isnan(row['city']):
                tags.append('city:%d' % row['city'])

            if type(row['gender']) == str:
                tags.append('gender:%s' % row['gender'])

            return tags

        path_users_id2idx = '%s/data-user2idx.json' % self.artifacts_dir
        path_songs_id2idx = '%s/data-song2idx.json' % self.artifacts_dir
        path_interactions = '%s/data-interactions-trn.npz' % self.artifacts_dir
        path_userfeatures = '%s/data-userfeatures.npz' % self.artifacts_dir
        path_songfeatures = '%s/data-songfeatures.npz' % self.artifacts_dir

        def read_return():
            with open(path_users_id2idx, 'r') as fp:
                uid2idx = json.load(fp)
            with open(path_songs_id2idx, 'r') as fp:
                sid2idx = json.load(fp)
            UF = load_npz(path_userfeatures)
            SF = load_npz(path_songfeatures)
            if train:
                TRN = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['msno', 'song_id', 'target'])
                UI = [uid2idx.get(x) for x in TRN['msno']]
                SI = [sid2idx.get(x) for x in TRN['song_id']]
                II = load_npz(path_interactions)
                return UI, SI, TRN['target'].values, II, UF, SF

            if test:
                assert True == False, "Implement Me!"

        if exists(path_interactions) and \
                exists(path_userfeatures) and \
                exists(path_songfeatures) and \
                exists(path_users_id2idx) and \
                exists(path_songs_id2idx):
            self.logger.info('Features already computed, reading and returning them.')
            return read_return()

        t0 = time()
        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['msno', 'song_id', 'target'])
        TST = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id', 'msno', 'song_id'])
        SNG = pd.read_csv('%s/songs.csv' % self.data_dir,
                          usecols=['song_id', 'artist_name', 'genre_ids', 'language'])
        SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
        MMB = pd.read_csv('%s/members.csv' % self.data_dir, usecols=['msno', 'city', 'bd', 'gender'])

        self.logger.info('Merge SNG and SEI.')
        SNG = SNG.merge(SEI, on='song_id', how='left')

        self.logger.info('Merge TRN and TST with SNG and MMB.')
        TRN = TRN.merge(MMB, on='msno', how='left')
        TST = TST.merge(MMB, on='msno', how='left')
        TRN = TRN.merge(SNG, on='song_id', how='left')
        TST = TST.merge(SNG, on='song_id', how='left')

        # Throw away unused after merging.
        del SEI, MMB

        # Combine train and test for tagging.
        TRNTST = TRN.append(TST)

        # Encoders for user and song IDs.
        users_id2idx = {x: i for i, x in enumerate(sorted(TRNTST['msno'].unique()))}
        songs_id2idx = {x: i for i, x in enumerate(sorted(TRNTST['song_id'].unique()))}

        with open(path_users_id2idx, 'w') as fp:
            json.dump(users_id2idx, fp)

        with open(path_songs_id2idx, 'w') as fp:
            json.dump(songs_id2idx, fp)

        self.logger.info('Tagging songs')
        df_songs = TRNTST.drop_duplicates(['song_id'])
        songs_ids = df_songs['song_id']
        songs_tags = df_songs.apply(song_to_tags, axis=1).values.tolist()
        songs_tag2idx = {x: i for i, x in enumerate(sorted(set(flatten(songs_tags))))}

        self.logger.info('Building songs CSR matrix')
        rows, cols = [], []
        for song_id, song_tags in zip(songs_ids, songs_tags):
            rows += [songs_id2idx[song_id]] * len(song_tags)
            cols += [songs_tag2idx[x] for x in song_tags]

        data = np.ones(len(rows), dtype=np.uint8)
        song_features = coo_matrix((data, (rows, cols))).tocsr()
        save_npz(path_songfeatures, song_features)
        self.logger.info('Saved %s (%d seconds)' % (path_songfeatures, time() - t0))

        self.logger.info('Tagging users')
        df_users = TRNTST.drop_duplicates(['msno'])
        users_ids = df_users['msno']
        users_tags = df_users.apply(user_to_tags, axis=1).values.tolist()
        users_tag2idx = {x: i for i, x in enumerate(sorted(set(flatten(users_tags))))}

        self.logger.info('Building users CSR matrix')
        rows, cols = [], []
        for user_id, user_tags in zip(users_ids, users_tags):
            rows += [users_id2idx[user_id]] * len(user_tags)
            cols += [users_tag2idx[x] for x in user_tags]

        data = np.ones(len(rows), dtype=np.uint8)
        user_features = coo_matrix((data, (rows, cols))).tocsr()
        save_npz(path_userfeatures, user_features)
        self.logger.info('Saved %s (%d seconds)' % (path_userfeatures, time() - t0))

        self.logger.info('Building interaction COO matrices')
        rows = [users_id2idx.get(x) for x in TRN['msno']]
        cols = [songs_id2idx.get(x) for x in TRN['song_id']]
        save_npz(path_interactions, coo_matrix((TRN['target'], (rows, cols))))
        self.logger.info('Saved %s (%d seconds)' % (path_interactions, time() - t0))

        return read_return()

    def fit(self):

        _, _, _, II, UF, SF = self.get_features(train=True)
        II_trn, II_val = random_train_test_split(II, test_percentage=0.2)

        model = LightFM(no_components=20, loss='logistic', learning_rate=0.1)

        for i in range(10):
            t0 = time()
            model.fit_partial(II_trn, user_features=UF, item_features=SF, num_threads=cpu_count(), verbose=False)
            yp_trn = model.predict(II_trn.row, II_trn.col, item_features=SF, user_features=UF, num_threads=cpu_count())
            yp_val = model.predict(II_val.row, II_val.col, item_features=SF, user_features=UF, num_threads=cpu_count())
            auc_trn = roc_auc_score(II_trn.data, yp_trn)
            auc_val = roc_auc_score(II_val.data, yp_val)
            self.logger.info('Epoch %d: AUC trn = %.3lf, AUC val = %.3lf (%.2lf seconds)' %
                             (i, auc_trn, auc_val, time() - t0))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = LFMRec(
        data_dir='data',
        artifacts_dir='artifacts/lfmrec',
    )

    if args['fit']:
        model.fit()

    elif args['predict']:
        model.predict()
