from glob import glob
from hashlib import sha256
from math import ceil
from scipy.misc import imread
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb

from keras.layers import Input, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, concatenate, dot, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K


def hashid(id_): return sha256(id_.encode()).hexdigest()


class UserSongConvEmbedding(object):

    def __init__(self,
                 nb_users=30755,
                 embed_size=200,
                 spec_time=1000,
                 spec_freq=128,
                 epochs=10,
                 batch_size=10,
                 optimizer_args={'lr': 0.0001, 'decay': 1e-6}):

        self.nb_users = nb_users
        self.embed_size = embed_size
        self.spec_time = spec_time
        self.spec_freq = spec_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_args = optimizer_args

    def fit(self, user_ids, spec_paths, targets):

        print(len(user_ids))
        print(np.mean(targets))

        gen_trn = self._sampler(user_ids[:-200], spec_paths[:-200], targets[:-200])
        gen_val = self._sampler(user_ids[-200:], spec_paths[-200:], targets[-200:])
        net, _ = self.network(self.spec_time, self.spec_freq, self.embed_size, self.nb_users)
        net.summary()
        net.compile(loss='binary_crossentropy',
                    optimizer=Adam(**self.optimizer_args),
                    metrics=['accuracy'])
        net.fit_generator(gen_trn,
                          steps_per_epoch=ceil(len(user_ids) / self.batch_size),
                          epochs=self.epochs,
                          validation_data=gen_val,
                          validation_steps=200 // self.batch_size,
                          verbose=1)

    def _sampler(self, user_ids, spec_paths, targets):

        Xs = np.zeros((self.batch_size, self.spec_time, self.spec_freq), dtype=np.float32)
        user_ids = np.array(user_ids)[:, np.newaxis]
        targets = np.array(targets)[:, np.newaxis]
        ii = np.arange(len(user_ids))

        while True:
            np.random.shuffle(ii)
            for ii_ in np.array_split(ii, ceil(len(ii) / self.batch_size)):
                if len(ii_) < self.batch_size:
                    continue
                for i, p in enumerate([spec_paths[x] for x in ii_]):
                    im = imread(p)
                    if im.shape[1] < self.spec_time:
                        im = np.tile(im, ceil(self.spec_time / im.shape[1]))
                    Xs[i] = im.T[:self.spec_time, :self.spec_freq] / 255.
                yield [user_ids[ii_], Xs], targets[ii_]

    @staticmethod
    def network(spec_time, spec_freq, embed_size, nb_users):

        # Input for user ID and song spectrogram.
        input_user = Input(shape=(1,))
        input_song = Input(shape=(spec_time, spec_freq))

        # Build the convolutional layers, based on Deep Content-based Music Recommendation (2014).
        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(input_song)
        x = MaxPooling1D(4)(x)
        x = LeakyReLU()(x)

        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        x = Conv1D(512, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        global_max = GlobalMaxPooling1D()(x)
        global_avg = GlobalAveragePooling1D()(x)
        x = concatenate([global_max, global_avg])

        x = Dense(2048)(x)
        x = LeakyReLU()(x)
        x = emb_song = out_song = Dense(embed_size)(x)

        # Song-only network.
        net_song = Model(input_song, out_song)

        # User embeddings.
        emb_users = Embedding(nb_users, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_user = emb_users(input_user)
        emb_user = Lambda(lambda x: K.squeeze(x, 1))(emb_user)

        # Dot product the song and user embeddings.
        dot_song_user = dot([emb_song, emb_user], axes=-1, name='dot')

        # Classify with a sigmoid activation.
        classify = Activation('sigmoid', name='classify')(dot_song_user)

        # Combined network.
        net_user_song = Model([input_user, input_song], classify)

        return net_user_song, net_song


if __name__ == "__main__":

    DATA_DIR = '/mnt/data/datasets/kkbox-scraping'
    TRN = pd.read_csv('data/train.csv', usecols=['msno', 'song_id', 'target'], nrows=15000)

    # Get the subset of training data that have spectrograms saved.
    specs_paths = glob('%s/*_melspec.jpg' % DATA_DIR)
    specs_id_hashes = set([x.split('/')[-1].replace('_melspec.jpg', '') for x in specs_paths])
    users_msno2id = {x: i for i, x in enumerate(TRN['msno'].unique())}
    ii_keep = [i for i, row in TRN.iterrows() if hashid(row['song_id']) in specs_id_hashes]
    TRN = TRN.ix[ii_keep]

    # Populate training data.
    user_ids, spec_paths = [], []
    for i, row in tqdm(TRN.iterrows()):
        user_ids.append(users_msno2id[row['msno']])
        spec_paths.append('%s/%s_melspec.jpg' % (DATA_DIR, hashid(row['song_id'])))

    # Train.
    model = UserSongConvEmbedding()
    model.fit(user_ids, spec_paths, TRN['target'].values)

    pdb.set_trace()

    pass
