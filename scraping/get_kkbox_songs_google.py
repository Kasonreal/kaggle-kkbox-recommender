from bs4 import BeautifulSoup as soup
from google import search
from hashlib import sha256
from os.path import exists
from os import remove
from urllib.request import urlretrieve
import pandas as pd
import pdb
import requests
import sys

DATA_DIR = 'data/kkbox_songs'
p = 'data/songs_listened_only.csv'
if not exists(p):
    TRN = pd.read_csv('data/train.csv')
    TST = pd.read_csv('data/test.csv')
    S = pd.read_csv('data/songs_merged.csv')
    S = S[S['song_id'].isin(set(TRN['song_id']).union(TST['song_id']))]
    S.to_csv(p)
else:
    S = pd.read_csv(p)

for i, row in S.iterrows():

    h = sha256(row['song_id'].encode()).hexdigest()
    url_path = '%s/%s_url.txt' % (DATA_DIR, h)
    htm_path = '%s/%s_page.html' % (DATA_DIR, h)
    mp3_path = '%s/%s_sample.mp3' % (DATA_DIR, h)
    print(row)
    print(url_path)
    print(htm_path)
    print(mp3_path)

    try:

        # Determine the KKBOX song URL via Google search, save to .txt file.
        if not exists(url_path):
            q = 'site:kkbox.com song %s %s' % (row['artist_name'], row['name'])
            u = [x for x in search(q, num=1, stop=1)][0]
            with open(url_path, 'w') as f:
                f.write('%s\n' % u)
        else:
            with open(url_path, 'r') as f:
                u = f.read().strip()
        print(u)

        # Scrape and save the KKBOX page.
        if not exists(htm_path):
            r = requests.get(u)
            h = r.content.decode()
            with open(htm_path, 'w') as f:
                f.write(h)
        else:
            with open(htm_path, 'r') as f:
                h = f.read()

        # Extract the sample URL from the KKBOX page and download it.
        if not exists(mp3_path):
            s = soup(h, 'html.parser')
            m = s.find('meta', property='music:preview_url:url')
            if m:
                urlretrieve(m['content'], mp3_path)
            else:
                print('No MP3 URL')

    # Delete all the files on a keyboard interrupt.
    except KeyboardInterrupt as e:
        if exists(url_path):
            remove(url_path)
        if exists(htm_path):
            remove(htm_path)
        if exists(mp3_path):
            remove(mp3_path)
        raise e

    print('%d / %d done' % (i, len(S)))
