import pandas as pd
from hashlib import sha256
from os.path import exists


def hashid(id_):
    return sha256(id_.encode()).hexdigest()


if __name__ == "__main__":

    TRN = pd.read_csv('data/train.csv')
    TST = pd.read_csv('data/test.csv')

    # Assign each training user a unique index.
    msno2index = {m: i + 1 for i, m in enumerate(TRN['msno'].unique())}

    for df, path in [(TRN, 'data/train_extra.csv'), (TST, 'data/test_extra.csv')]:

        # Hash the song IDs to get the ID used for scraping.
        df['song_id_hash'] = [hashid(x) for x in df['song_id']]
        print(df.shape)

        # Spectrogram paths.
        df['spec_path'] = ['data/kkbox-melspecs/%s_melspec.jpg' % x for x in df['song_id_hash']]
        print(df.shape)

        # Whether or not the spectrogram has been downloaded.
        df['spec_ready'] = [exists(x) for x in df['spec_path']]
        print(df.shape)

        # User index used for user embeddings.
        # Only users included in the training set have an embedding.
        df['user_index'] = [msno2index[m] if m in msno2index else 0 for m in df['msno']]
        print(df.shape)

        df.to_csv(path, index=False)
        print('Saved %s' % path)
