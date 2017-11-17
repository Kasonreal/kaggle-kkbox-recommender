from glob import glob
from more_itertools import flatten, chunked
from multiprocessing import Pool, cpu_count
from os import getenv
from os.path import exists
from time import time
from tqdm import tqdm
import argparse
import ffm as ffmlib
import gc
import json
import logging
import numpy as np
import pandas as pd
import pdb
import pickle
import sys

from hyperopt import hp, fmin, tpe
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

np.random.seed(865)

MISSING_TOKEN = 'xxxxxxxx'


class FFMClassifier():

    def __init__(self,
                 factor_size=33,
                 learning_rate=0.09,
                 reg=0.0001,
                 nb_epochs_max=2):

        self.factor_size = factor_size
        self.learning_rate = learning_rate
        self.reg = reg
        self.nb_epochs_max = nb_epochs_max

    def fit(self, X_trn, y_trn, X_val, y_val, model_path=None):
        logger = logging.getLogger(str(self))
        ffm = ffmlib.FFM(eta=self.learning_rate, lam=self.reg, k=self.factor_size)
        ffm_data_trn = ffmlib.FFMData(X_trn, y_trn)
        ffm_data_val = ffmlib.FFMData(X_val, y_val)
        ffm.init_model(ffm_data_trn)
        auc_trn_max = auc_val_max = auc_val = nb_epochs = 0.
        while auc_val == auc_val_max and nb_epochs < self.nb_epochs_max:
            nb_epochs += 1
            ffm.iteration(ffm_data_trn)
            auc_trn = roc_auc_score(y_trn, ffm.predict(ffm_data_trn))
            auc_val = roc_auc_score(y_val, ffm.predict(ffm_data_val))
            logger.info('AUC trn: %.3lf AUC val: %.3lf' % (auc_trn, auc_val))
            auc_trn_max = max(auc_trn, auc_trn_max)
            auc_val_max = max(auc_val, auc_val_max)
            if auc_val == auc_val_max and model_path:
                logger.info('Saving %s' % model_path)
                ffm.save_model(model_path)
        del ffm_data_trn, ffm_data_val
        return auc_trn_max, auc_val_max, nb_epochs

    def predict(self, X, model_path):
        ffm = ffmlib.read_model(model_path)
        ffm_data = ffmlib.FFMData(X, np.zeros(len(X)))
        yp = ffm.predict(ffm_data)
        del ffm_data
        return yp

    def __repr__(self):
        return '%s: %d, %.4lf, %.4lf' % \
            (self.__class__.__name__, self.factor_size, self.learning_rate, self.reg)


def cols_to_fields_series(series):
    return [
        (0, series['u_idx'], 1),
        (1, series['s_art'], 1),
        (2, series['u_cit'], 1),
        (3, series['u_age'], 1),
        (4, series['u_gen'], 1),
        (5, series['s_idx'], 1),
        (6, series['s_lan'], 1),
        (7, series['s_gen'], 1),
        (8, series['s_yea'], 1),
        (9, series['s_cou'], 1)
    ]


def cols_to_fields_df(df):
    return df.apply(cols_to_fields_series, axis=1).values.tolist()


class FFMRec(object):

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 features_glob_trn,
                 features_glob_tst,
                 model_path,
                 predict_path_tst,
                 ffm_hyperparams):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_glob_trn = features_glob_trn
        self.features_glob_tst = features_glob_tst
        self.model_path = model_path
        self.predict_path_tst = predict_path_tst
        self.ffm_hyperparams = ffm_hyperparams
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_features(self, train=False, test=False):

        pptrn, pptst = glob(self.features_glob_trn), glob(self.features_glob_tst)
        if len(pptrn) and len(pptst):
            self.logger.info('Reading features from disk.')
            xtrn, ytrn, xtst = [], [], []
            if train:
                for p in tqdm(sorted(pptrn)):
                    with open(p, 'rb') as fp:
                        xtrn += pickle.load(fp)
                df = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['target'])
                ytrn = df['target'].values.tolist()
            if test:
                for p in tqdm(sorted(pptst)):
                    with open(p, 'rb') as fp:
                        xtst += pickle.load(fp)

            return xtrn, ytrn, xtst

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['msno', 'song_id', 'target'])
        TST = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id', 'msno', 'song_id'])
        SNG = pd.read_csv('%s/songs.csv' % self.data_dir,
                          usecols=['song_id', 'artist_name', 'genre_ids', 'language'])
        SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
        MMB = pd.read_csv('%s/members.csv' % self.data_dir, usecols=['msno', 'city', 'bd', 'gender'])

        self.logger.info('Merge SNG and SEI.')
        SNG = SNG.merge(SEI, on='song_id', how='left')

        self.logger.info('Merge TRN and TEST with SNG and MMB.')
        TRN = TRN.merge(MMB, on='msno', how='left')
        TST = TST.merge(MMB, on='msno', how='left')
        TRN = TRN.merge(SNG, on='song_id', how='left')
        TST = TST.merge(SNG, on='song_id', how='left')

        # Throw away unused after merging.
        del SEI, MMB
        gc.collect()

        # Impute with a common missing token.
        for c in TRN.columns[TRN.isnull().any()]:
            self.logger.info('Imputing %s' % c)
            TRN[c].fillna(MISSING_TOKEN, inplace=True)

        for c in TST.columns[TST.isnull().any()]:
            self.logger.info('Imputing %s' % c)
            TST[c].fillna(MISSING_TOKEN, inplace=True)

        assert len(TRN.columns[TRN.isnull().any()]) == 0
        assert len(TST.columns[TST.isnull().any()]) == 0

        def round_year(n, s=5):
            try:
                return round(int(n) / s) * s
            except (TypeError, ValueError):
                return n

        self.logger.info('Converting ISRC to year and country')
        TRN['country'] = TRN['isrc'].apply(lambda s: s[:2])
        TST['country'] = TST['isrc'].apply(lambda s: s[:2])
        TRN['year'] = TRN['isrc'].apply(lambda s: round_year(s[5:7]))
        TST['year'] = TST['isrc'].apply(lambda s: round_year(s[5:7]))
        TRN.drop('isrc', axis=1, inplace=True)
        TST.drop('isrc', axis=1, inplace=True)

        self.logger.info('Normalizing genres')
        sort_genres = lambda s: '|'.join(sorted(s.split('|')))
        TRN['genre_ids'] = TRN['genre_ids'].apply(sort_genres)
        TST['genre_ids'] = TST['genre_ids'].apply(sort_genres)

        self.logger.info('Normalizing artists')
        sort_artists = lambda s: '|'.join(sorted(s.lower().split('| ')))
        TRN['artist_name'] = TRN['artist_name'].apply(sort_artists)
        TST['artist_name'] = TST['artist_name'].apply(sort_artists)

        self.logger.info('Normalizing age')
        round_age = lambda n, s=5: round(min(max(n, 0), 50) / s) * s
        TRN['bd'] = TRN['bd'].apply(round_age)
        TST['bd'] = TST['bd'].apply(round_age)

        cc = [
            ('msno', 'u_idx', np.uint16),
            ('artist_name', 's_art', np.uint16),
            ('city', 'u_cit', np.uint8),
            ('bd', 'u_age', np.uint8),
            ('gender', 'u_gen', np.uint8),

            ('song_id', 's_idx', np.uint32),
            ('language', 's_lan', np.uint8),
            ('genre_ids', 's_gen', np.uint16),
            ('year', 's_yea', np.uint8),
            ('country', 's_cou', np.uint8),
        ]
        for cold, cnew, dtype in cc:
            self.logger.info('Label encoding %s -> %s' % (cold, cnew))
            enc = LabelEncoder()
            enc.fit(TRN[cold].append(TST[cold]).apply(str))
            TRN[cnew] = enc.transform(TRN[cold].apply(str)).astype(dtype)
            TST[cnew] = enc.transform(TST[cold].apply(str)).astype(dtype)
            TRN.drop(cold, axis=1, inplace=True)
            TST.drop(cold, axis=1, inplace=True)
            assert cold not in TRN.columns
            assert cold not in TST.columns

        def chunk_convert_save(df, glob_str, chunk_size=100000):
            pool = Pool(cpu_count())
            nb_chunks = len(df) // chunk_size + 1
            df_chunks = np.array_split(df, nb_chunks)
            for i in tqdm(range(nb_chunks)):
                x = pool.map(cols_to_fields_df, np.array_split(df_chunks[i], cpu_count()))
                p = glob_str.replace('*', '%03d' % i)
                with open(p, 'wb') as fp:
                    pickle.dump(list(flatten(x)), fp)
            pool.close()

        self.logger.info('Converting train to FFM format')
        chunk_convert_save(TRN, self.features_glob_trn)

        self.logger.info('Converting test to FFM format')
        chunk_convert_save(TST, self.features_glob_tst)

    def hpo(self, n_splits=4, seed=423, evals=10):

        X, y, _ = self.get_features(train=True)

        # Hack semi-global variable.
        self._auc_max = 0.

        def obj(args):
            args['factor_size'] = int(args['factor_size'])
            model = FFMClassifier(**args)
            self.logger.info('%s: %d samples' % (str(model), len(X)))
            kfold = StratifiedKFold(n_splits=n_splits, random_state=seed)
            aucs, epcs = [], []
            for ii_trn, ii_val in kfold.split(X, y):
                auc_trn, auc_val, epochs = model.fit(
                    [X[i] for i in ii_trn], [y[i] for i in ii_trn],
                    [X[i] for i in ii_val], [y[i] for i in ii_val])
                aucs.append(auc_val)
                epcs.append(epochs)
                auc_mean = np.mean(aucs)
                epc_mean = np.mean(epcs)
                self.logger.info('%.3lf %.3lf' % (auc_mean, epc_mean))

            self.logger.info('%s: %.3lf, %.3lf' % (str(model), epc_mean, auc_mean))
            if auc_mean > self._auc_max:
                self.logger.info('*' * len(str(args)))
                self.logger.info('AUC improved from %.3lf to %.3lf' % (self._auc_max, auc_mean))
                self.logger.info(args)
                self._auc_max = auc_mean
                self.logger.info('*' * len(str(args)))
            del model
            return -1 * auc_mean

        space = {
            'factor_size': hp.uniform('factor_size', 5, 50),
            'learning_rate': hp.uniform('learning_rate', 0.09, 0.11),
        }
        best = fmin(obj, space=space, algo=tpe.suggest, max_evals=evals, rstate=np.random.RandomState(int(time())))
        self.logger.info(best, self._auc_max)

    def fit(self):
        self.logger.info('Reading features from disk')
        X, y, _ = self.get_features(train=True)
        model = FFMClassifier(**self.ffm_hyperparams)
        self.logger.info(str(model))
        model.fit(X, y, X, y, model_path=self.model_path)

    def predict(self):
        df = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id'])
        _, _, X = self.get_features(test=True)
        model = ffmlib.read_model(self.model_path)
        ffm_tst = ffmlib.FFMData(X, np.zeros(len(X)))
        df['target'] = model.predict(ffm_tst)
        self.logger.info('Mean target: %.3lf' % df['target'].mean())
        df.to_csv(self.predict_path_tst, index=False)
        self.logger.info('Saved %s' % self.predict_path_tst)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--hpo', action='store_true', default=False)
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = FFMRec(
        data_dir='data',
        artifacts_dir='artifacts/ffmrec',
        features_glob_trn='artifacts/ffmrec/features_trn_*.pkl',
        features_glob_tst='artifacts/ffmrec/features_tst_*.pkl',
        model_path='artifacts/ffmrec/ffm_best.bin',
        predict_path_tst='artifacts/ffmrec/predict_tst_%d.csv' % int(time()),
        ffm_hyperparams={}
    )

    model.get_features()

    if args['hpo']:
        model.hpo()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()
