# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from __future__ import print_function
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import os

# [195] feature importance
#       msno                                     1556798.984
#       source_type                              795306.337
#       count_song_played                        476090.797
#       artist_name                              299454.501
#       source_screen_name                       130773.790
#       song_id                                  74350.807
#       source_system_tab                        65517.758
#       expiration_year                          40592.008
#       membership_days                          39486.011
#       composer                                 36042.020
#       count_artist_played                      26260.791
#       genre_ids                                18868.822
#       song_year                                16691.620
#       expiration_date                          14244.976
#       registration_date                        13847.002
#       bd                                       11551.204
#       lyricist                                 10844.090
#       registration_month                       9806.185
#       expiration_month                         9719.590
#       city                                     7994.302
#       language                                 7867.750
#       song_length                              5455.277
#       registered_via                           5273.702
#       registration_year                        4068.269
#       gender                                   3458.008
#       composer_count                           1599.858
#       song_lang_boolean                        1334.710
#       lyricists_count                          916.432
#       genre_ids_count                          397.539
#       artist_count                             249.582
#       artist_composer                          113.109
#       smaller_song                             88.130
#       is_featured                              39.408
#       artist_composer_lyricist                 19.422
# [200]   trn's auc: 0.819148     val's auc: 0.680013

if not os.path.exists('artifacts/vikas/train.csv'):

    print('Loading data...')
    data_path = 'data/'
    train = pd.read_csv(data_path + 'train.csv',
                        dtype={'msno': 'category',
                               'source_system_tab': 'category',
                               'source_screen_name': 'category',
                               'source_type': 'category',
                               'target': np.uint8,
                               'song_id': 'category'})
    test = pd.read_csv(data_path + 'test.csv',
                       dtype={'msno': 'category',
                              'source_system_tab': 'category',
                              'source_screen_name': 'category',
                              'source_type': 'category',
                              'song_id': 'category'})
    songs = pd.read_csv(data_path + 'songs.csv',
                        dtype={'genre_ids': 'category',
                               'language': 'category',
                               'artist_name': 'category',
                               'composer': 'category',
                               'lyricist': 'category',
                               'song_id': 'category'})
    members = pd.read_csv(data_path + 'members.csv',
                          dtype={'city': 'category',
                                 'bd': np.uint8,
                                 'gender': 'category',
                                 'registered_via': 'category'},
                          parse_dates=['registration_init_time', 'expiration_date'])
    songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
    print('Done loading...')

    print('Data merging...')
    train = train.merge(songs, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    members['membership_days'] = members['expiration_date'].subtract(
        members['registration_init_time']).dt.days.astype(int)
    members['registration_year'] = members['registration_init_time'].dt.year
    members['registration_month'] = members['registration_init_time'].dt.month
    members['registration_date'] = members['registration_init_time'].dt.day
    members['expiration_year'] = members['expiration_date'].dt.year
    members['expiration_month'] = members['expiration_date'].dt.month
    members['expiration_date'] = members['expiration_date'].dt.day
    members = members.drop(['registration_init_time'], axis=1)

    def isrc_to_year(isrc):
        if type(isrc) == str:
            if int(isrc[5:7]) > 17:
                return 1900 + int(isrc[5:7])
            else:
                return 2000 + int(isrc[5:7])
        else:
            return np.nan

    songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
    songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)
    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')
    train = train.merge(songs_extra, on='song_id', how='left')
    train.song_length.fillna(200000, inplace=True)
    train.song_length = train.song_length.astype(np.uint32)
    train.song_id = train.song_id.astype('category')
    test = test.merge(songs_extra, on='song_id', how='left')
    test.song_length.fillna(200000, inplace=True)
    test.song_length = test.song_length.astype(np.uint32)
    test.song_id = test.song_id.astype('category')
    del members, songs
    gc.collect()
    print('Done merging...')

    print("Adding new features")

    def genre_id_count(x):
        if x == 'no_genre_id':
            return 0
        else:
            return x.count('|') + 1

    train['genre_ids'].fillna('no_genre_id', inplace=True)
    test['genre_ids'].fillna('no_genre_id', inplace=True)
    train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
    test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)

    def lyricist_count(x):
        if x == 'no_lyricist':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        return sum(map(x.count, ['|', '/', '\\', ';']))

    train['lyricist'].fillna('no_lyricist', inplace=True)
    test['lyricist'].fillna('no_lyricist', inplace=True)
    train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
    test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

    def composer_count(x):
        if x == 'no_composer':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

    train['composer'].fillna('no_composer', inplace=True)
    test['composer'].fillna('no_composer', inplace=True)
    train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
    test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

    def is_featured(x):
        if 'feat' in str(x):
            return 1
        return 0

    train['artist_name'].fillna('no_artist', inplace=True)
    test['artist_name'].fillna('no_artist', inplace=True)
    train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
    test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)

    def artist_count(x):
        if x == 'no_artist':
            return 0
        else:
            return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

    train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
    test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

    # if artist is same as composer
    train['artist_composer'] = (train['artist_name'] == train['composer']).astype(np.int8)
    test['artist_composer'] = (test['artist_name'] == test['composer']).astype(np.int8)

    # if artist, lyricist and composer are all three same
    train['artist_composer_lyricist'] = ((train['artist_name'] == train['composer']) & (
        train['artist_name'] == train['lyricist']) & (train['composer'] == train['lyricist'])).astype(np.int8)
    test['artist_composer_lyricist'] = ((test['artist_name'] == test['composer']) & (
        test['artist_name'] == test['lyricist']) & (test['composer'] == test['lyricist'])).astype(np.int8)

    # is song language 17 or 45.

    def song_lang_boolean(x):
        if '17.0' in str(x) or '45.0' in str(x):
            return 1
        return 0

    train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
    test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)

    _mean_song_length = np.mean(train['song_length'])

    def smaller_song(x):
        if x < _mean_song_length:
            return 1
        return 0

    train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
    test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

    # number of times a song has been played before
    _dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
    _dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}

    def count_song_played(x):
        try:
            return _dict_count_song_played_train[x]
        except KeyError:
            try:
                return _dict_count_song_played_test[x]
            except KeyError:
                return 0

    train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
    test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

    # number of times the artist has been played
    _dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
    _dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}

    def count_artist_played(x):
        try:
            return _dict_count_artist_played_train[x]
        except KeyError:
            try:
                return _dict_count_artist_played_test[x]
            except KeyError:
                return 0

    train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
    test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)

    print("Done adding features")
    train.to_csv('artifacts/vikas/train.csv', index=False)
    test.to_csv('artifacts/vikas/test.csv', index=False)


train = pd.read_csv('artifacts/vikas/train.csv')
test = pd.read_csv('artifacts/vikas/test.csv')
nb_trn = int(len(train) * 0.8)

print("Train test and validation sets")
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X = train.drop(['target'], axis=1)
y = train['target'].values

lgb_trn = lgb.Dataset(X.iloc[:nb_trn], y[:nb_trn])
lgb_val = lgb.Dataset(X.iloc[nb_trn:], y[nb_trn:])

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.3,
    'verbose': 0,
    'num_leaves': 108,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 10,
    'num_rounds': 200,
    'metric': 'auc'
}


def print_feature_importance(print_every_iterations=10, importance_type='gain'):

    def callback(env):
        if env.iteration % print_every_iterations != 0:
            return
        names = env.model.feature_name()
        ivals = env.model.feature_importance(importance_type)
        print('[%d] feature importance' % env.iteration)
        p = len('[%d]' % env.iteration)
        for i in np.argsort(-1 * ivals):
            print('%s %-40s %.3lf' % (' ' * p, names[i], ivals[i]))

    callback.order = 99
    return callback

cb = [print_feature_importance(5)]
model_f1 = lgb.train(params, train_set=lgb_trn,  valid_sets=[lgb_trn, lgb_val],
                     valid_names=['trn', 'val'], verbose_eval=5, callbacks=cb)
