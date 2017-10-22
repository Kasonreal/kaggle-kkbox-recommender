from glob import glob
from hashlib import sha256
from math import ceil
from scipy.misc import imread
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb

np.random.seed(865)

from keras.layers import Input, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, concatenate, dot, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
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

        net = self.network(self.embed_size, self.nb_users, self.nb_songs, self.nb_artists)
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

        net = Model([inp_user, inp_song, inp_arti], [clf_user_song, clf_user_arti])
        return net


if __name__ == "__main__":

    TRN = pd.read_csv('data/train_fe.csv', usecols=['user_index', 'song_index', 'artist_index', 'target'])

    # Train.
    model = USAE(nb_users=TRN['user_index'].max(),
                 nb_songs=TRN['song_index'].max(),
                 nb_artists=TRN['artist_index'].max())
    model.fit(TRN['user_index'].values,
              TRN['song_index'].values,
              TRN['artist_index'].values,
              TRN['target'].values)

    pdb.set_trace()

    pass
