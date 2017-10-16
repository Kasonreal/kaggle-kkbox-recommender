from bs4 import BeautifulSoup as soup
from hashlib import sha256
from os.path import exists
from os import remove, mkdir
from urllib.request import urlretrieve
from sys import argv
import pandas as pd
import pdb
import requests

# Parallel scraping script designed to be run on multiple EC2 instances.
# Start and end points for parallel scraping.
assert len(argv) == 3
i0 = int(argv[1])
i1 = int(argv[2])

# Read the merged songs and keep only those that are
# used in the training and testing datasets.
merged_path = 'data/songs_merged.csv'
scrape_path = 'data/songs_scrape.csv'
if not exists(scrape_path):
    print('Filtering songs for scraping...')
    TRN = pd.read_csv('data/train.csv')
    TST = pd.read_csv('data/test.csv')
    S = pd.read_csv(merged_path)
    S = S[S['song_id'].isin(set(TRN['song_id']).union(TST['song_id']))]
    S.to_csv(scrape_path)
else:
    S = pd.read_csv(scrape_path)

# Make directory for storing KKBOX data.
kkbox_data_dir = 'data/kkbox_songs'
if not exists(kkbox_data_dir):
    mkdir(kkbox_data_dir)

# Begin scraping.
SEARCH_BASE = 'https://www.kkbox.com/sg/en/search.php?word='

for i, row in S.iloc[i0:i1].iterrows():

    # Use the hash of the song id as the ID.
    # Many of the song_ids are not file-name friendly.
    h = sha256(row['song_id'].encode()).hexdigest()
    url_path = '%s/%s_url.txt' % (kkbox_data_dir, h)
    htm_path = '%s/%s_page.html' % (kkbox_data_dir, h)
    mp3_path = '%s/%s_sample.mp3' % (kkbox_data_dir, h)
    print(row)
    print(url_path)
    print(htm_path)
    print(mp3_path)

    # Search KKBOX directly. Given the search results, always pick the first.
    if not exists(url_path):
        search_url = '%s%s %s' % (SEARCH_BASE, row['artist_name'], row['name'])
        print('Search URL %s' % search_url)
        req = requests.get(search_url)
        htm = soup(req.content, 'html.parser')
        song_item = htm.select('table.song-table tr.song-item')[0]
        song_url = song_item.select('a.song-title')[0]['href']
        song_url = 'https://www.kkbox.com%s' % song_url
        print('Song URL %s' % song_url)
        with open(url_path, 'w') as f:
            f.write('%s\n' % song_url)

    else:
        with open(url_path, 'r') as f:
            song_url = f.read().strip()

    # Scrape and save the KKBOX page.
    if not exists(htm_path):
        print('Requesting %s' % song_url)
        req = requests.get(song_url)
        htm = req.content.decode()
        with open(htm_path, 'w') as f:
            f.write(htm)
    else:
        with open(htm_path, 'r') as f:
            htm = f.read()

    # Extract the sample URL from the KKBOX page and download it.
    if not exists(mp3_path):
        htm = soup(htm, 'html.parser')
        print('Page title: %s' % htm.title)
        mp3_meta = htm.find('meta', property='music:preview_url:url')
        if mp3_meta:
            mp3_url = mp3_meta['content']
            print('Requesting %s' % mp3_url)
            urlretrieve(mp3_url, mp3_path)
        else:
            print('No MP3 URL')

    print('%d / %d done' % (i, len(S)))
    print('*' * 30)
