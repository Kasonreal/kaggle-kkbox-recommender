# Based on exmaple from:
# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
# Mel-spectrogram recommended by http://benanne.github.io/2014/08/05/spotify-cnns.html.
# It's dimensions are much lower than a regular spectrogram e.g. matplotlib.specgram
from multiprocessing import Pool
from scipy.misc import imsave
from glob import glob
from tqdm import tqdm
import librosa
import numpy as np
import pdb


def mel_parallel(args):
    DATA_DIR, id_ = args
    p_mp3 = '%s/%s_sample.mp3' % (DATA_DIR, id_)
    p_jpg = '%s/%s_melspec.jpg' % (DATA_DIR, id_)
    try:
        y, sr = librosa.load(p_mp3, res_type='kaiser_fast')
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        imsave(p_jpg, S_db)
    except ValueError as e:
        print(id_, e)
        pass


if __name__ == "__main__":

    DATA_DIR = '/mnt/data/datasets/kkbox-scraping'
    pp_mp3 = glob('%s/*_sample.mp3' % DATA_DIR)
    pp_jpg = glob('%s/*_melspec.jpg' % DATA_DIR)
    id_mp3 = set([x.split('/')[-1].replace('_sample.mp3', '') for x in pp_mp3])
    id_mel = set([x.split('/')[-1].replace('_melspec.jpg', '') for x in pp_jpg])
    id_todo = list(id_mp3 - id_mel)

    nb_parallel = 8
    pool = Pool(nb_parallel)
    for i in tqdm(range(0, len(id_todo), nb_parallel)):
        args = [(DATA_DIR, id_) for id_ in id_todo[i:i + nb_parallel]]
        res = pool.map(mel_parallel, args)
