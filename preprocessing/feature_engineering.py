from hashlib import sha256
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pdb

if __name__ == "__main__":

    TRN = pd.read_csv('data/train.csv')
    TST = pd.read_csv('data/test.csv')
    SNG = pd.read_csv('data/songs.csv')

    # Merge train, test with songs.
    TRN = pd.merge(TRN, SNG, on='song_id')
    TST = pd.merge(TST, SNG, on='song_id')

    # Hash the song_id for dealing with scraped data.
    TRN['song_id_hash'] = [sha256(x.encode()).hexdigest() for x in TRN['song_id']]
    TST['song_id_hash'] = [sha256(x.encode()).hexdigest() for x in TST['song_id']]

    # Impute missing values with the most common value.
    impute_cols = [
        'source_system_tab',
        'source_screen_name',
        'source_type',
        'genre_ids',
        'language',
    ]
    for c in impute_cols:
        cmb = TRN[c].append(TST[c])
        val = cmb.value_counts().idxmax()
        TRN[c].fillna(val, inplace=True)
        TST[c].fillna(val, inplace=True)

    # Convert song length into seconds.
    TRN['song_length_sec'] = TRN['song_length'] / 1000.
    TST['song_length_sec'] = TST['song_length'] / 1000.

    # Encode a subset of the columns.
    encode_cols = [
        ('msno', 'user_index'),
        ('song_id', 'song_index'),
        ('artist_name', 'artist_index'),
        ('source_system_tab', 'source_system_tab_index'),
        ('source_screen_name', 'source_screen_name_index'),
        ('source_type', 'source_type_index'),
        ('language', 'language_index'),
    ]
    for ca, cb in encode_cols:
        cmb = TRN[ca].append(TST[ca])
        enc = LabelEncoder()
        enc.fit(cmb)
        TRN[cb] = enc.transform(TRN[ca])
        TST[cb] = enc.transform(TST[ca])

    # Add columns that will be populated.
    TRN['dist_user_song'] = np.zeros(len(TRN))
    TST['dist_user_song'] = np.zeros(len(TST))
    TRN['dist_user_artist'] = np.zeros(len(TRN))
    TST['dist_user_artist'] = np.zeros(len(TST))

    # Keep a subset of all the columns.
    keep_cols_trn = [
        'user_index',
        'song_index',
        'artist_index',
        'source_system_tab_index',
        'source_screen_name_index',
        'source_type_index',
        'language_index',
        'song_length_sec',
        'dist_user_song',
        'dist_user_artist',
        'target'
    ]
    keep_cols_tst = ['id'] + keep_cols_trn[:-1]
    TRN = TRN[keep_cols_trn]
    TST = TST[keep_cols_tst]

    print(TRN.isnull().sum())
    print(TST.isnull().sum())

    # Save.
    TRN.to_csv('data/train_fe.csv')
    TST.to_csv('data/test_fe.csv')
