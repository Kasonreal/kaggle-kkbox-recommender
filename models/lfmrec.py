from itertools import product
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from multiprocessing import cpu_count
from more_itertools import flatten
from os.path import exists
from pprint import pformat
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from time import time
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pickle
import pdb
import sys

NTH = cpu_count()
np.random.seed(865)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LFMRec(object):
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

        def round_to(n, r):
            return round(n / r) * r

        def song_to_tags(row):
            tags = []

            tags.append('song:%s' % row['song_id'])

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

        def user_to_tags(row):
            tags = []

            tags.append('user:%s' % row['msno'])
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
            assert train or test
            UF = load_npz(path_userfeatures)
            SF = load_npz(path_songfeatures)
            if train:
                II = load_npz(path_interactions)
                return II, UF, SF
            if test:
                TST = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['msno', 'song_id'])
                with open(path_users_id2idx, 'r') as fp:
                    users_id2idx = json.load(fp)
                with open(path_songs_id2idx, 'r') as fp:
                    songs_id2idx = json.load(fp)
                UI = np.array([users_id2idx[x] for x in TST['msno']])
                SI = np.array([songs_id2idx[x] for x in TST['song_id']])
                return UI, SI, UF, SF

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

    def hpo(self):

        II, UF, SF = self.get_features(train=True)

        hpgrid = list(product(
            [10**x for x in range(-6, -4)] + [0],
            ['adagrad', 'adadelta'],
            [10**x for x in range(-4, 0)],
            range(10, 160, 20),
        ))

        _rng = np.random.RandomState(int(time()))
        _rng.shuffle(hpgrid)

        epochs_max = 100
        nb_folds = 5
        skfolds = StratifiedKFold(nb_folds)
        auc_val_mean_max = 0
        best_params = None

        for l2, opt, lr, nc in hpgrid:
            params = {'no_components': nc,
                      'learning_rate': lr,
                      'item_alpha': l2,
                      'user_alpha': l2,
                      'learning_schedule': opt}
            self.logger.info('\n%s' % pformat(params))
            auc_val_mean = epochs_mean = 0
            for i, (ii_trn, ii_val) in enumerate(skfolds.split(II.data, II.data)):
                self.logger.info('Fold %d / %d' % (i + 1, nb_folds))
                model = LightFM(loss='logistic', **params)
                II_trn = coo_matrix((II.data[ii_trn], (II.row[ii_trn], II.col[ii_trn])))
                II_val = coo_matrix((II.data[ii_val], (II.row[ii_val], II.col[ii_val])))
                auc_val = auc_val_max = epochs = ltmax = 0
                while ltmax < 3 and epochs < epochs_max:
                    t0 = time()
                    model.fit_partial(II_trn, user_features=UF, item_features=SF, num_threads=NTH)
                    yp_trn = model.predict(II_trn.row, II_trn.col, SF, UF, num_threads=NTH)
                    yp_val = model.predict(II_val.row, II_val.col, SF, UF, num_threads=NTH)
                    auc_trn = roc_auc_score(II_trn.data, sigmoid(yp_trn))
                    auc_val = roc_auc_score(II_val.data, sigmoid(yp_val))
                    auc_val_max = max(auc_val_max, auc_val)
                    ltmax = 0 if auc_val == auc_val_max else ltmax + 1
                    self.logger.info('Epoch %d: trn = %.3lf val = %.3lf (%d sec)' %
                                     (epochs, auc_trn, auc_val, time() - t0))
                    epochs += 1
                auc_val_mean += auc_val_max / nb_folds
                epochs_mean += (epochs - ltmax) / nb_folds

            self.logger.info('AUC mean = %.3lf' % (auc_val_mean))

            if auc_val_mean >= auc_val_mean_max:
                auc_val_mean_max = auc_val_mean
                best_params = params
                params['epochs'] = round(epochs_mean, 3)

            self.logger.info('-' * 80)
            self.logger.info('Best AUC mean so far = %.3lf' % auc_val_mean_max)
            self.logger.info('Best params so far:\n%s' % pformat(best_params))
            self.logger.info('-' * 80)

            fp = open('%s/search-params.txt' % self.artifacts_dir, 'w')
            fp.write('%.3lf\n%s\n' % (auc_val_mean_max, pformat(best_params)))
            fp.close()

    def fit(self):

        II, UF, SF = self.get_features(train=True)
        model = LightFM(no_components=100, loss='logistic', learning_rate=0.1, item_alpha=1e-4, user_alpha=1e-4)

        # II_trn, II_val = random_train_test_split(II, test_percentage=0.2, random_state=np.random)
        # for i in range(100):
        #     t0 = time()
        #     model.fit_partial(II_trn, user_features=UF, item_features=SF, num_threads=cpu_count(), verbose=False)
        #     yp_trn = model.predict(II_trn.row, II_trn.col, item_features=SF, user_features=UF, num_threads=cpu_count())
        #     yp_val = model.predict(II_val.row, II_val.col, item_features=SF, user_features=UF, num_threads=cpu_count())
        #     auc_trn = roc_auc_score(II_trn.data, sigmoid(yp_trn))
        #     auc_val = roc_auc_score(II_val.data, sigmoid(yp_val))
        #     self.logger.info('Epoch %d: AUC trn = %.3lf, AUC val = %.3lf (%.2lf seconds)' %
        #                      (i, auc_trn, auc_val, time() - t0))

        for i in range(4):
            t0 = time()
            model.fit_partial(II, user_features=UF, item_features=SF, num_threads=NTH, verbose=False)
            yp = sigmoid(model.predict(II.row, II.col, item_features=SF, user_features=UF, num_threads=NTH))
            auc = roc_auc_score(II.data, yp)
            self.logger.info('Epoch %d: AUC trn = %.3lf (%.2lf seconds)' % (i, auc, time() - t0))
        with open(self.best_model_path, 'wb') as fp:
            pickle.dump(model, fp)
            self.logger.info('Saved model to %s' % self.best_model_path)

    def predict(self):

        UI, SI, UF, SF = self.get_features(test=True)

        with open(self.best_model_path, 'rb') as fp:
            model = pickle.load(fp)

        yp, b = np.zeros(len(UI), dtype=np.float64), 100000
        for i in range(0, len(UI), b):
            yp[i:i + b] = model.predict(UI[i:i + b], SI[i:i + b], SF, UF, num_threads=NTH)

        yp = sigmoid(yp)
        self.logger.info('Target mean: %.3lf' % np.mean(yp))

        df = pd.DataFrame({'id': list(range(len(yp))), 'target': yp})
        df.to_csv(self.predict_path, index=False)
        self.logger.info('Saved %s' % self.predict_path)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--hpo', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = LFMRec(
        data_dir='data',
        artifacts_dir='artifacts/lfmrec',
        best_model_path='artifacts/lfmrec/model_best.pkl',
        predict_path='artifacts/lfmrec/predict_tst_%d.csv' % int(time())
    )

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()

    if args['hpo']:
        model.hpo()
