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


class USCE(object):
    "User-song-conv embeddings."

    def __init__(self,
                 nb_users=30755,
                 embed_size=200,
                 spec_time=1000,
                 spec_freq=128,
                 epochs=10,
                 batch_size=200,
                 val_split=0.75,
                 optimizer_args={'lr': 0.0005, 'decay': 1e-6}):

        self.nb_users = nb_users
        self.embed_size = embed_size
        self.spec_time = spec_time
        self.spec_freq = spec_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.optimizer_args = optimizer_args

    def fit(self, user_indexes, spec_paths, targets):

        def trg_true(yt, yp):
            return K.mean(yt)

        def trg_pred(yt, yp):
            return K.mean(K.round(yp))

        nb_trn = int(len(user_indexes) * self.val_split)
        nb_val = len(user_indexes) - nb_trn
        gen_trn = self._sampler(user_indexes[:nb_trn], spec_paths[:nb_trn], targets[:nb_trn])
        gen_val = self._sampler(user_indexes[-nb_val:], spec_paths[-nb_val:], targets[-nb_val:])
        net, _ = self.network(self.spec_time, self.spec_freq, self.embed_size, self.nb_users)
        net.summary()
        net.compile(loss='binary_crossentropy',
                    optimizer=Adam(**self.optimizer_args),
                    metrics=['accuracy', trg_true, trg_pred])

        print('Training on %d samples' % nb_trn)
        print('Validating on %d samples' % nb_val)

        cb = [
            ModelCheckpoint('artifacts/usce_{val_loss:.2f}.hdf5',
                            monitor='val_loss',
                            save_best_only=True,
                            verbose=1,
                            mode='min'),
            CSVLogger('artifacts/usce_logs.csv'),
        ]

        net.fit_generator(gen_trn,
                          steps_per_epoch=nb_trn // self.batch_size,
                          validation_data=gen_val,
                          validation_steps=nb_val // self.batch_size,
                          epochs=self.epochs,
                          verbose=1,
                          callbacks=cb)

    def _sampler(self, user_indexes, spec_paths, targets):

        Xs = np.zeros((self.batch_size, self.spec_time, self.spec_freq), dtype=np.float32)
        user_indexes = np.array(user_indexes)[:, np.newaxis]
        targets = np.array(targets)[:, np.newaxis]
        ii = np.arange(len(user_indexes))

        while True:
            np.random.shuffle(ii)
            for i in range(0, len(ii) // self.batch_size * self.batch_size, self.batch_size):
                ii_ = ii[i:i + self.batch_size]
                for i, p in enumerate([spec_paths[x] for x in ii_]):
                    im = imread(p)
                    if im.shape[1] < self.spec_time:
                        im = np.tile(im, ceil(self.spec_time / im.shape[1]))
                    Xs[i] = im.T[:self.spec_time, :self.spec_freq] / 255.

                yield [user_indexes[ii_], Xs], targets[ii_]

    @staticmethod
    def network(spec_time, spec_freq, embed_size, nb_users):

        # Input for user ID and song spectrogram.
        input_user = Input(shape=(1,))
        input_song = Input(shape=(spec_time, spec_freq))

        # Build the convolutional layers, based on Deep Content-based Music Recommendation (2014).
        x = Dropout(0.2)(input_song)
        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(4)(x)
        x = LeakyReLU()(x)

        x = Dropout(0.2)(x)
        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        x = Dropout(0.2)(x)
        x = Conv1D(512, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        global_max = GlobalMaxPooling1D()(x)
        global_avg = GlobalAveragePooling1D()(x)
        x = concatenate([global_max, global_avg])

        x = Dropout(0.2)(x)
        x = Dense(1024)(x)
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
    TRN = pd.read_csv('data/train_extra.csv')

    # Get subset of training data that have spectrograms saved.
    TRN = TRN[TRN['spec_ready'] == True]

    # Train.
    model = USCE(nb_users=TRN['user_index'].max() + 1)
    model.fit(TRN['user_index'].values, TRN['spec_path'].values, TRN['target'].values)

    pdb.set_trace()

    pass
