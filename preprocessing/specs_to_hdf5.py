# Convert all spectrograms into a single hdf5 file.
# Missing songs are replaced by a spectrogram from the same artist.
from hashlib import sha256
from math import ceil
from scipy.misc import imread
from os.path import exists
from tqdm import tqdm
import h5py
import json
import numpy as np
import pandas as pd
import pdb

if __name__ == "__main__":

    SNG = pd.read_csv('data/songs_scrape.csv')
    path_template = 'data/kkbox-melspecs/{hash:s}_melspec.jpg'

    sfp = h5py.File('data/melspecs.hdf5', 'w')
    spec_time = 1000
    spec_freq = 128
    specs = sfp.create_dataset('specs', (len(SNG), spec_time, spec_freq), dtype='uint8')
    song_id_to_index = dict()

    # For handling missing spectrograms. Missing spectrograms are replaced
    # by a spectrogram from the same artist. Spectrograms smaller than the
    # desired size are tiled across time.
    artist_to_index = {x: [] for x in SNG['artist_name']}
    missing = []
    nb_tiled = 0

    for i, (_, row) in tqdm(enumerate(SNG.iterrows())):
        song_id = row['song_id']
        artist_name = row['artist_name']
        path = path_template.format(hash=sha256(song_id.encode()).hexdigest())
        song_id_to_index[song_id] = i
        if not exists(path):
            missing.append((i, artist_name))
            continue
        im = imread(path)
        if im.shape[1] < spec_time:
            im = np.tile(im, ceil(spec_time / im.shape[1]))
            nb_tiled += 1
        specs[i, :, :] = im.T[:spec_time, :spec_freq]
        artist_to_index[artist_name].append(i)

    sfp.attrs['song_id_to_index'] = json.dumps(song_id_to_index)

    for i, artist_name in tqdm(missing):
        if len(artist_to_index[artist_name]) > 0:
            j = artist_to_index[artist_name][0]
        else:
            j = artist_to_index['Various Artists'][0]
        specs[i, :, :] = specs[j, :, :]

    print('Stored: %d' % specs.shape[0])
    print('Tiled: %d' % nb_tiled)
    print('Missing: %d' % len(missing))

    sfp.close()
