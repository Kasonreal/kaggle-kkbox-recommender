from glob import glob
from hashlib import sha256
from math import ceil
from scipy.misc import imread
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import pdb

np.random.seed(865)

from keras.layers import Input, Embedding, Activation, dot, Lambda
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K


class USAE(object):
    """User, song, artist embeddings.
    cos(user, song) = target
    cos(user, artist) = target
    cos(song, artist) = 1
    """

    def __init__(self,
                 nb_users,
                 nb_songs,
                 nb_artists,
                 embed_size=100,
                 epochs=30,
                 batch_size=20000,
                 optimizer_args={'lr': 0.01, 'decay': 1e-4}):

        self.nb_users = nb_users
        self.nb_songs = nb_songs
        self.nb_artists = nb_artists
        self.embed_size = embed_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_args = optimizer_args

    def fit(self, user_indexes, song_indexes, artist_indexes, targets):

        net, _ = self.network(self.embed_size, self.nb_users, self.nb_songs, self.nb_artists)
        net.summary()
        net.compile(loss='binary_crossentropy',
                    optimizer=Adam(**self.optimizer_args),
                    metrics=['accuracy'])

        print('%d users, %d songs, %d artists' % (
            user_indexes.max(), song_indexes.max(), artist_indexes.max()))

        X = [user_indexes, song_indexes, artist_indexes]
        Y = [targets[:, np.newaxis], targets[:, np.newaxis]]

        cb = [
            ModelCheckpoint('artifacts/usae_%d_{loss:.2f}.hdf5' % self.embed_size,
                            monitor='loss',
                            save_best_only=True,
                            verbose=1,
                            mode='min'),
            CSVLogger('artifacts/usae_logs.csv'),
        ]

        net.fit(X, Y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                shuffle=True,
                callbacks=cb)

    @staticmethod
    def network(embed_size, nb_users, nb_songs, nb_artists):

        inp_user = Input(shape=(1,))
        inp_song = Input(shape=(1,))
        inp_arti = Input(shape=(1,))

        emb_users = Embedding(nb_users, embed_size, name='embed_users',
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_user = emb_users(inp_user)

        emb_songs = Embedding(nb_songs, embed_size, name='embed_songs',
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_song = emb_songs(inp_song)

        emb_artis = Embedding(nb_artists, embed_size, name='embed_artis',
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_arti = emb_artis(inp_arti)

        dot_user_song = dot([emb_user, emb_song], axes=-1)
        dot_user_song = Lambda(lambda x: K.squeeze(x, 1))(dot_user_song)
        dot_user_arti = dot([emb_user, emb_arti], axes=-1)
        dot_user_arti = Lambda(lambda x: K.squeeze(x, 1))(dot_user_arti)

        clf_user_song = Activation('sigmoid', name='user_song')(dot_user_song)
        clf_user_arti = Activation('sigmoid', name='user_arti')(dot_user_arti)

        # First network used for training.
        net_train = Model([inp_user, inp_song, inp_arti], [clf_user_song, clf_user_arti])

        # Second network used to compute similarities.
        sim_user_song = dot([emb_user, emb_song], axes=-1, normalize=True)
        sim_user_song = Lambda(lambda x: K.squeeze(x, 1))(sim_user_song)
        sim_user_arti = dot([emb_user, emb_arti], axes=-1, normalize=True)
        sim_user_arti = Lambda(lambda x: K.squeeze(x, 1))(sim_user_arti)

        net_simil = Model([inp_user, inp_song, inp_arti], [sim_user_song, sim_user_arti])

        return net_train, net_simil


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--model_path', default='artifacts/usae_100_0.48.hdf5')
    ap.add_argument('--train', action='store_true', default=False)
    ap.add_argument('--similarities', action='store_true', default=False)
    args = vars(ap.parse_args())

    TRN = pd.read_csv('data/train_fe.csv')

    if args['train']:

        model = USAE(nb_users=TRN['user_index'].max(),
                     nb_songs=TRN['song_index'].max(),
                     nb_artists=TRN['artist_index'].max())
        model.fit(TRN['user_index'].values,
                  TRN['song_index'].values,
                  TRN['artist_index'].values,
                  TRN['target'].values)

    if args['similarities']:

        model = USAE(nb_users=TRN['user_index'].max(),
                     nb_songs=TRN['song_index'].max(),
                     nb_artists=TRN['artist_index'].max())
        _, net = model.network(model.embed_size, model.nb_users, model.nb_songs, model.nb_artists)
        net.load_weights(args['model_path'], by_name=True)

        A = TRN[TRN['target'] == 0]
        B = TRN[TRN['target'] == 1]

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots(1, 2)

        X = [A['user_index'].values, A['song_index'].values, A['artist_index'].values]
        sim_user_song, sim_user_arti = net.predict(X, batch_size=model.batch_size, verbose=1)
        print(np.mean(sim_user_song), np.mean(sim_user_arti))
        fig.axes[0].hist(sim_user_song, label='user-song 0', bins=50, alpha=0.3, color='blue')
        fig.axes[1].hist(sim_user_arti, label='user-arti 0', bins=50, alpha=0.3, color='red')

        X = [B['user_index'].values, B['song_index'].values, B['artist_index'].values]
        sim_user_song, sim_user_arti = net.predict(X, batch_size=model.batch_size, verbose=1)
        print(np.mean(sim_user_song), np.mean(sim_user_arti))
        fig.axes[0].hist(sim_user_song, label='user-song 1', bins=50, alpha=0.3, color='green')
        fig.axes[1].hist(sim_user_arti, label='user-arti 1', bins=50, alpha=0.3, color='purple')
        fig.axes[0].legend()
        fig.axes[1].legend()
        plt.savefig('hist.png')

        pdb.set_trace()
