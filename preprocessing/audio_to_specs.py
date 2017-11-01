# Based on exmaple from:
# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
# Mel-spectrogram recommended by http://benanne.github.io/2014/08/05/spotify-cnns.html.
# It's dimensions are much lower than a regular spectrogram e.g. matplotlib.specgram
from multiprocessing import Pool
from scipy.misc import imsave
from glob import glob
from os import listdir
from time import time, sleep
from tqdm import tqdm
import librosa
import numpy as np
import pdb
import sys
import threading

READ_DIR = '/mnt/data/datasets/kkbox-scraping'
WRIT_DIR = 'data/kkbox-melspecs'


def monitor(nb_all):
    t0 = time()
    nb_start = len(listdir(WRIT_DIR))
    while True:
        pp = listdir(WRIT_DIR)
        elapsed = time() - t0
        rate = (len(pp) - nb_start) / elapsed
        remaining = (nb_all - len(pp)) / max(rate, 1)
        print('Completed: %d / %d, %.5lf' % (len(pp), nb_all, len(pp) / nb_all))
        print('Rate: %.3lf / second' % rate)
        print('Hours elapsed: %.3lf' % (elapsed / 60 / 60))
        print('Hours remaining: %.3lf' % (remaining / 60 / 60))
        print('*' * 30)
        if len(pp) == nb_all:
            return
        sleep(15)


def mel_parallel(id_todo):

    for id_ in id_todo:
        p_mp3 = '%s/%s_sample.mp3' % (READ_DIR, id_)
        p_jpg = '%s/%s_melspec.jpg' % (WRIT_DIR, id_)
        try:
            y, sr = librosa.load(p_mp3, res_type='kaiser_fast')
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            imsave(p_jpg, S_db)
        except ValueError as e:
            print(id_, e)
            pass


if __name__ == "__main__":

    assert len(sys.argv) == 2
    nb_parallel = int(sys.argv[1])

    print('Computing missing IDs')
    pp_mp3 = glob('%s/*_sample.mp3' % READ_DIR)
    pp_jpg = glob('%s/*_melspec.jpg' % WRIT_DIR)
    id_mp3 = set([x.split('/')[-1].replace('_sample.mp3', '') for x in pp_mp3])
    id_mel = set([x.split('/')[-1].replace('_melspec.jpg', '') for x in pp_jpg])
    id_todo = list(id_mp3 - id_mel)

    monitor_thread = threading.Thread(target=monitor, args=(len(id_mp3),))
    monitor_thread.start()

    pool = Pool(nb_parallel)
    res = pool.map(mel_parallel, np.array_split(id_todo, nb_parallel))

    sys.exit(0)
