from collections import Counter
from glob import glob
from os.path import exists
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse
import dill
import json
import logging
import numpy as np
import pandas as pd
import pdb

np.random.seed(865)

from keras.layers import Input, Embedding, Activation, dot, Lambda
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping, ReduceLROnPlateau
from keras import backend as K

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

LAYER_NAME_USERS = 'EMBED_USERS'
LAYER_NAME_SONGS = 'EMBED_SONGS'
LAYER_NAME_ARTISTS = 'EMBED_ARTISTS'

NB_USERS = 34403
NB_SONGS = 419839
NB_ARTISTS = 46372


class EmbeddingSummary(Callback):
    """Compute, display summary statistics about embeddings"""

    def __init__(self, net_sim, X, Y, hist_path):
        self.net_sim = net_sim
        self.X = X
        self.Y = Y
        self.hist_path = hist_path

    def on_epoch_end(self, epoch, logs):
        logger = logging.getLogger('EmbeddingSummary')
        us, ua = self.net_sim.predict(self.X, batch_size=self.params['batch_size'] * 10, verbose=1)

        usn = us[np.where(self.Y[0] == 0)]
        usp = us[np.where(self.Y[0] == 1)]
        uan = ua[np.where(self.Y[1] == 0)]
        uap = ua[np.where(self.Y[1] == 1)]

        logger.info('\n')
        logger.info('Mean user-song negatives:   %.4lf' % usn.mean())
        logger.info('Mean user-song positives:   %.4lf (%.4lf)' % (usp.mean(), usp.mean() - usn.mean()))
        logger.info('Mean user-artist negatives: %.4lf' % uan.mean())
        logger.info('Mean user-artist positives: %.4lf (%.4lf)' % (uap.mean(), uap.mean() - uan.mean()))

        fig, _ = plt.subplots(1, 2)
        fig.axes[0].hist(usn, alpha=0.3, color='blue', label='user-song negatives')
        fig.axes[0].hist(usp, alpha=0.3, color='red', label='user-song positives')
        fig.axes[0].legend()
        fig.axes[1].hist(uan, alpha=0.3, color='blue', label='uaer-artist negatives')
        fig.axes[1].hist(uap, alpha=0.3, color='red', label='usar-artist positives')
        fig.axes[1].legend()
        plt.savefig(self.hist_path)


class CFSC(object):
    """
    Content-Free Similarity Classifier.
    Classifier that uses user, song, and artist similarities without
    explicitly processing any content."""

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 features_path_trn,
                 features_path_tst,
                 embedding_path,
                 classifier_path,
                 predict_path_trn,
                 predict_path_tst,
                 embedding_size,
                 embedding_epochs,
                 embedding_batch,
                 embedding_steps,
                 embedding_optimizer_args):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_path_trn = features_path_trn
        self.features_path_tst = features_path_tst
        self.embedding_path = embedding_path
        self.classifier_path = classifier_path
        self.predict_path_trn = predict_path_trn
        self.predict_path_tst = predict_path_tst
        self.embedding_size = embedding_size
        self.embedding_epochs = embedding_epochs
        self.embedding_batch = embedding_batch
        self.embedding_steps = embedding_steps
        self.embedding_optimizer_args = embedding_optimizer_args
        self.logger = logging.getLogger('CFSC')

    def get_features(self):

        self.logger.info('Computing features')
        if exists(self.features_path_trn) and exists(self.features_path_tst):
            self.logger.info('Features already computed')
            return

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir)
        TST = pd.read_csv('%s/test.csv' % self.data_dir)
        SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
        SNG = SNG[['song_id', 'song_length', 'artist_name']]

        self.logger.info('Replacing missing song IDs')
        missing_song_ids = set(TRN['song_id'].append(TST['song_id'])) - set(SNG['song_id'])
        SNG = SNG.append(pd.DataFrame({
            'song_id': list(missing_song_ids),
            'song_length': [SNG['song_length'].mean()] * len(missing_song_ids),
            'artist_name': [SNG['artist_name'].value_counts().idxmax()] * len(missing_song_ids)
        }))

        self.logger.info('Merging train, test with songs')
        TRN = TRN.merge(SNG, on='song_id', how='left')
        TST = TST.merge(SNG, on='song_id', how='left')

        # Impute missing values with the most common value.
        impute_cols = [
            'source_system_tab',
            'source_screen_name',
            'source_type'
        ]
        for c in impute_cols:
            self.logger.info('Imputing %s' % c)
            cmb = TRN[c].append(TST[c])
            val = cmb.value_counts().idxmax()
            TRN[c].fillna(val, inplace=True)
            TST[c].fillna(val, inplace=True)

        # Convert song length into minutes.
        TRN['song_length'] = TRN['song_length'] / 1000. / 60.
        TST['song_length'] = TST['song_length'] / 1000. / 60.

        # Encode a subset of the columns.
        encode_cols = [
            ('msno', 'user_index'),
            ('song_id', 'song_index'),
            ('artist_name', 'artist_index'),
            ('source_system_tab', 'source_system_tab_index'),
            ('source_screen_name', 'source_screen_name_index'),
            ('source_type', 'source_type_index'),
        ]
        for ca, cb in encode_cols:
            self.logger.info('Encoding %s -> %s' % (ca, cb))
            cmb = TRN[ca].append(TST[ca])
            enc = LabelEncoder()
            enc.fit(cmb)
            TRN[cb] = pd.Series(enc.transform(TRN[ca]), dtype='category')
            TST[cb] = pd.Series(enc.transform(TST[ca]), dtype='category')

        # Add similarity columns that will be populated.
        TRN['sim_user_song'] = np.zeros(len(TRN))
        TST['sim_user_song'] = np.zeros(len(TST))
        TRN['sim_user_artist'] = np.zeros(len(TRN))
        TST['sim_user_artist'] = np.zeros(len(TST))

        # Keep a subset of all the columns.
        self.logger.info('Removing unused columns')
        keep_cols_trn = [
            'user_index',
            'song_index',
            'artist_index',
            'source_system_tab_index',
            'source_screen_name_index',
            'source_type_index',
            'song_length',
            'sim_user_song',
            'sim_user_artist',
            'target'
        ]
        keep_cols_tst = ['id'] + keep_cols_trn[:-1]
        TRN = TRN[keep_cols_trn]
        TST = TST[keep_cols_tst]

        # TODO: Find "similar" users, songs, and artists for those that
        # are in the testing set but not in the training set. Until
        # this is done, those users, songs, and artists will be random.

        # self.logger.info('Replacing missing users, songs, artists with random non-missing ones')
        # missing_users = set(TST['user_index']) - set(TRN['user_index'])
        # missing_songs = set(TST['song_index']) - set(TRN['song_index'])
        # missing_artists = set(TST['artist_index']) - set(TRN['artist_index'])
        # ix = TST[TST['user_index'].isin(missing_users)].index
        # TST.ix[ix, 'user_index'] = TRN['user_index'].values[:len(ix)]
        # ix = TST[TST['song_index'].isin(missing_songs)].index
        # TST.ix[ix, 'song_index'] = TRN['song_index'].values[:len(ix)]
        # ix = TST[TST['artist_index'].isin(missing_artists)].index
        # TST.ix[ix, 'artist_index'] = TRN['artist_index'].values[:len(ix)]

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

    def _embedding_sampler(self, TRN):

        # Frequency of the user, song, and artist at each row.
        user_freqs = TRN.groupby(['user_index'])['user_index'].transform('count').values
        song_freqs = TRN.groupby(['song_index'])['song_index'].transform('count').values

        # Probability distribution for sampling with inverse probability of users and songs.
        # Popular users and songs are sampled less frequently than in-frequent ones.
        user_probs = 1. / user_freqs
        user_probs /= user_probs.sum()
        song_probs = 1. / song_freqs
        song_probs /= song_probs.sum()

        C1, C2 = Counter(), Counter()

        # Yield batches sampled based on user and song probabilities.
        # Alternate yielding samples from the user and song distributions.
        ii = np.arange(len(TRN))
        while True:
            uii = np.random.choice(ii, self.embedding_steps * self.embedding_batch, p=user_probs)
            sii = np.random.choice(ii, self.embedding_steps * self.embedding_batch, p=song_probs)

            for u in TRN['user_index'][uii]:
                C1[u] += 1

            for s in TRN['song_index'][sii]:
                C2[s] += 1

            x = list(C1.values())
            print(np.min(x), np.max(x), np.median(x))
            x = list(C2.values())
            print(np.min(x), np.max(x), np.median(x))

            # with open('users.json', 'w') as fp:
            #     json.dump(C1, fp, indent=2)
            #
            # with open('songs.json', 'w') as fp:
            #     json.dump(C2, fp, indent=2)

            for i in range(0, self.embedding_steps * self.embedding_batch, self.embedding_batch):
                for bii in [uii[i:i + self.embedding_batch], sii[i:i + self.embedding_batch]]:
                    X = [TRN['user_index'][bii].values, TRN['song_index'][bii].values, TRN['artist_index'][bii].values]
                    Y = [TRN['target'][bii].values, TRN['target'][bii].values]
                    yield X, Y

    def fit_embedding(self):

        net_trn, net_sim = self._networks(self.embedding_size, NB_USERS, NB_SONGS, NB_ARTISTS)
        net_trn.summary()
        net_trn.compile(loss='binary_crossentropy',
                        optimizer=Adam(**self.embedding_optimizer_args),
                        metrics=['accuracy'])

        self.logger.info('%d users, %d songs, %d artists' % (NB_USERS, NB_SONGS, NB_ARTISTS))

        TRN = pd.read_csv(self.features_path_trn)
        gen_trn = self._embedding_sampler(TRN)

        cb = [
            ModelCheckpoint(self.embedding_path,
                            monitor='loss',
                            save_best_only=True,
                            verbose=1,
                            mode='min'),
            CSVLogger('artifacts/usae_logs.csv'),
            EarlyStopping(monitor='loss', patience=5, min_delta=0.002),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, epsilon=0.005, min_lr=0.0001),
            # EmbeddingSummary(net_sim, X, Y, '%s/hist.png' % self.artifacts_dir),
        ]

        net_trn.fit_generator(gen_trn,
                              steps_per_epoch=self.embedding_steps,
                              epochs=self.embedding_epochs,
                              callbacks=cb,
                              verbose=1)

    def fit_classifier(self):

        # Setup network.
        _, net_sim = self._networks(self.embedding_size, NB_USERS, NB_SONGS, NB_ARTISTS)
        net_sim.load_weights(self.embedding_path, by_name=True)

        # Compute and populate similarity features.
        TRN = pd.read_csv(self.features_path_trn)
        X = [TRN['user_index'], TRN['song_index'], TRN['artist_index']]
        us, ua = net_sim.predict(X, batch_size=100000, verbose=1)
        TRN['sim_user_song'] = us[:, 0]
        TRN['sim_user_artist'] = ua[:, 0]
        print('\n')

        # Train a classifier.
        features_trn = [
            'source_system_tab_index',
            'source_screen_name_index',
            'source_type_index',
            'song_length',
            'sim_user_song',
            'sim_user_artist',
        ]

        X, Y = TRN[features_trn], TRN['target']
        X_trn, X_val, Y_trn, Y_val = train_test_split(X, Y, test_size=0.1, random_state=np.random)

        self.logger.info('Training Classifier')
        clf = RandomForestClassifier(verbose=2, random_state=np.random, n_jobs=4)
        # clf = DecisionTreeClassifier()
        clf.fit(X_trn, Y_trn)

        Y_prd = clf.predict(X_trn)
        self.logger.info('Train accuracy: %.4lf' % accuracy_score(Y_trn, Y_prd))
        self.logger.info('Train roc-auc:  %.4lf' % roc_auc_score(Y_trn, Y_prd))

        Y_prd = clf.predict(X_val)
        self.logger.info('Validation accuracy: %.4lf' % accuracy_score(Y_val, Y_prd))
        self.logger.info('Validation roc-auc:  %.4lf' % roc_auc_score(Y_val, Y_prd))

        with open(self.classifier_path, 'wb') as fp:
            dill.dump((clf, features_trn), fp)
        self.logger.info('Saved %s' % self.classifier_path)

    def predict_classifier(self):

        # Load classifier and features.
        with open(self.classifier_path, 'rb') as fp:
            clf, features_trn = dill.load(fp)

        # Setup network.
        _, net_sim = self._networks(self.embedding_size, NB_USERS, NB_SONGS, NB_ARTISTS)
        net_sim.load_weights(self.embedding_path, by_name=True)

        # Compute and populate similarity features.
        TST = pd.read_csv(self.features_path_tst)
        X = [TST['user_index'], TST['song_index'], TST['artist_index']]
        us, ua = net_sim.predict(X, batch_size=100000, verbose=1)
        TST['sim_user_song'] = us[:, 0]
        TST['sim_user_artist'] = ua[:, 0]

        # Make predictions.
        X = TST[features_trn]
        PRD = pd.DataFrame(data={'id': TST['id'], 'target': clf.predict(X)})
        PRD.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Rows: %d' % len(PRD))
        self.logger.info('Target mean: %.4lf' % PRD['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)

    @staticmethod
    def _networks(embed_size, nb_users, nb_songs, nb_artists):

        inp_user = Input(shape=(1,))
        inp_song = Input(shape=(1,))
        inp_arti = Input(shape=(1,))

        emb_users = Embedding(nb_users, embed_size, name=LAYER_NAME_USERS,
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_user = emb_users(inp_user)

        emb_songs = Embedding(nb_songs, embed_size, name=LAYER_NAME_SONGS,
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_song = emb_songs(inp_song)

        emb_artis = Embedding(nb_artists, embed_size, name=LAYER_NAME_ARTISTS,
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_arti = emb_artis(inp_arti)

        dot_user_song = dot([emb_user, emb_song], axes=-1)
        dot_user_arti = dot([emb_user, emb_arti], axes=-1)

        clf_user_song = Activation('sigmoid')(dot_user_song)
        clf_user_song = Lambda(lambda x: K.squeeze(x, -1), name='user_song')(clf_user_song)
        clf_user_arti = Activation('sigmoid')(dot_user_arti)
        clf_user_arti = Lambda(lambda x: K.squeeze(x, -1), name='user_arti')(clf_user_arti)

        # First network used for training.
        net_trn = Model([inp_user, inp_song, inp_arti], [clf_user_song, clf_user_arti])

        # Second network used to compute similarities.
        sim_user_song = dot([emb_user, emb_song], axes=-1, normalize=True)
        sim_user_song = Lambda(lambda x: K.squeeze(x, 1))(sim_user_song)
        sim_user_arti = dot([emb_user, emb_arti], axes=-1, normalize=True)
        sim_user_arti = Lambda(lambda x: K.squeeze(x, 1))(sim_user_arti)

        net_sim = Model([inp_user, inp_song, inp_arti], [sim_user_song, sim_user_arti])

        return net_trn, net_sim


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit_embedding', action='store_true', default=False)
    ap.add_argument('--fit_classifier', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = CFSC(
        data_dir='data',
        artifacts_dir='artifacts/cfsc',
        features_path_trn='artifacts/cfsc/features_trn.csv',
        features_path_tst='artifacts/cfsc/features_tst.csv',
        embedding_path='artifacts/cfsc/keras_embeddings_best.hdf5',
        classifier_path='artifacts/cfsc/classifier.pkl',
        predict_path_trn='artifacts/cfsc/predict_trn.csv',
        predict_path_tst='artifacts/cfsc/predict_tst.csv',
        embedding_size=100,
        embedding_epochs=50,
        embedding_batch=20000,
        embedding_steps=500,
        embedding_optimizer_args={'lr': 0.01, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit_embedding']:
        model.fit_embedding()

    if args['fit_classifier']:
        model.fit_classifier()

    if args['predict']:
        model.predict_classifier()
