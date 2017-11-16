from hashlib import md5
from math import ceil
from os import getenv
from os.path import exists
from time import time
from tqdm import tqdm
import argparse
import ffm as ffmlib
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


class FFMClassifier():

    def __init__(self,
                 factor_size=10,
                 learning_rate=0.1,
                 reg=0.0001):

        self.factor_size = factor_size
        self.learning_rate = learning_rate
        self.reg = reg

    def fit(self, X_trn, y_trn, X_val, y_val, nb_epochs_max=30, model_path=None):
        logger = logging.getLogger(str(self))
        ffm = ffmlib.FFM(eta=self.learning_rate, lam=self.reg, k=self.factor_size)
        ffm_data_trn = ffmlib.FFMData(X_trn, y_trn)
        ffm_data_val = ffmlib.FFMData(X_val, y_val)
        ffm.init_model(ffm_data_trn)
        auc_trn_max = auc_val_max = auc_val = nb_epochs = 0.
        while auc_val == auc_val_max and nb_epochs < nb_epochs_max:
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
        del ffm_data_trn
        del ffm_data_val
        return auc_trn_max, auc_val_max, nb_epochs

    def predict(self, X, model_path):
        ffm = ffmlib.read_model(model_path)
        ffm_data = ffmlib.FFMData(X, np.zeros(len(X)))
        yp = ffm.predict(ffm_data)
        del ffm_data
        return yp

    def __repr__(self):
        return '%s: %d, %.3lf, %.3lf' % \
            (self.__class__.__name__, self.factor_size, self.learning_rate, self.reg)


class FFMRec(object):

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 features_path_trn,
                 features_path_tst,
                 model_path,
                 predict_path_tst,
                 ffm_hyperparams):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_path_trn = features_path_trn
        self.features_path_tst = features_path_tst
        self.model_path = model_path
        self.predict_path_tst = predict_path_tst
        self.ffm_hyperparams = ffm_hyperparams
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_features(self):
        if exists(self.features_path_trn) and exists(self.features_path_tst):
            self.logger.info('Features already computed')
            return

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['msno', 'song_id', 'target'], nrows=100000)
        TST = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id', 'msno', 'song_id'], nrows=100000)

        cc = [('msno', 'u_idx'),
              ('song_id', 's_idx'), ]
        for cold, cnew in cc:
            self.logger.info('Label encoding %s -> %s' % (cold, cnew))
            enc = LabelEncoder()
            enc.fit(TRN[cold].append(TST[cold]).apply(str))
            TRN[cnew] = enc.transform(TRN[cold].apply(str))
            TST[cnew] = enc.transform(TST[cold].apply(str))
            TRN.drop(cold, axis=1, inplace=True)
            TST.drop(cold, axis=1, inplace=True)
            assert cold not in TRN.columns
            assert cold not in TST.columns

        # Looping over existing features to convert them into [(field, index, value), ..] format.
        # Missing data is simply left missing in this format.
        self.logger.info('Converting features to FFM format')

        def cols_to_fields(row):
            return [(i, row[c], 1) for i, (_, c) in enumerate(cc)]

        xtrn = TRN.apply(cols_to_fields, axis=1).values.tolist()
        xtst = TST.apply(cols_to_fields, axis=1).values.tolist()
        ytrn = TRN['target'].values.tolist()

        self.logger.info('Pickling, saving features')
        with open(self.features_path_trn, 'wb') as fp:
            pickle.dump((xtrn, ytrn), fp)
        self.logger.info('Saved %s with %d samples' % (self.features_path_trn, len(xtrn)))
        with open(self.features_path_tst, 'wb') as fp:
            pickle.dump((xtst, None), fp)
        self.logger.info('Saved %s with %d samples' % (self.features_path_tst, len(xtst)))

    def hpo(self, n_splits=4, seed=423, evals=10):

        self.logger.info('Reading features from disk')
        with open(self.features_path_trn, 'rb') as fp:
            X, y = pickle.load(fp)

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
            'factor_size': hp.uniform('factor_size', 4, 100),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'reg': hp.uniform('reg', 1e-5, 1e-4),
        }
        best = fmin(obj, space=space, algo=tpe.suggest, max_evals=evals, rstate=np.random.RandomState(int(time())))
        print(best, self._auc_max)

    def fit(self):
        self.logger.info('Reading features from disk')
        with open(self.features_path_trn, 'rb') as fp:
            X, y = pickle.load(fp)
        model = FFMClassifier(**self.ffm_hyperparams)
        self.logger.info(str(model))
        model.fit(X, y, X, y, self.model_path)

    def predict(self):
        df = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id'])
        with open(self.features_path_tst, 'rb') as fp:
            XTST, _ = pickle.load(fp)
        model = ffm.read_model(self.model_path)
        ffm_tst = ffm.FFMData(XTST, np.zeros(len(XTST)))
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
        features_path_trn='artifacts/ffmrec/features_trn.pkl',
        features_path_tst='artifacts/ffmrec/features_tst.pkl',
        model_path='artifacts/ffmrec/ffm_best.bin',
        predict_path_tst='artifacts/ffmrec/predict_tst_%d.csv' % int(time()),
        ffm_hyperparams={}
    )

    model.get_features()

    if args['hpo']:
        model.hpo()

    # if args['fit']:
    #     model.fit()

    # if args['predict']:
    #     model.predict()
