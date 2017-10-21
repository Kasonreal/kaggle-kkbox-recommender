from hashlib import sha256
from os.path import exists
from os import remove, mkdir
from pprint import pprint
from tqdm import tqdm
from time import time
import boto3
import numpy as np
import pandas as pd
import pdb
import random

API_URLS = [
    'https://ilx9qnkbo6.execute-api.ap-southeast-1.amazonaws.com/api/',
    'https://k1ke0a116c.execute-api.us-east-1.amazonaws.com/api/',
    'https://fmrtzlgxoh.execute-api.us-east-2.amazonaws.com/api/',
    'https://mypldsn3s7.execute-api.us-west-1.amazonaws.com/api/',
    'https://xvwk49vi2a.execute-api.us-west-2.amazonaws.com/api/'
]

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
    S.to_csv(scrape_path, index=False)
else:
    S = pd.read_csv(scrape_path)

# Fetch existing IDs from S3.
S3 = boto3.resource('s3')
B = S3.Bucket('kkbox-scraping')
oo = [o for o in tqdm(B.objects.all())]
s3_ids = set([o.key.split('_')[0] for o in oo if o.key.endswith('.mp3')])
print('%d downloads complete' % len(s3_ids))

# Import grequests after making S3 call because it messes with boto3.
import grequests

# Requests bookkeeping.
nb_reqs_total = 0
nb_success = 0
nb_reqs_batch = 300
reqs = []
req_times = []

# Start scraping missing songs.
for i, row in S.iterrows():

    song_id = sha256(row['song_id'].encode()).hexdigest()
    if song_id in s3_ids:
        continue

    if len(reqs) < nb_reqs_batch:
        j = {
            "song_id": song_id,
            "song_name": row['name'],
            "artist_name": row['artist_name'],
            "song_seconds": int(row['song_length'] / 1000)
        }
        u = '%sscrape' % random.choice(API_URLS)
        reqs.append(grequests.post(u, json=j))

    if len(reqs) == nb_reqs_batch or i + 1 == len(S):

        t0 = time()
        ress = grequests.map(reqs)
        req_times.append(time() - t0)
        for res in ress:
            if res == None:
                print(res)
            elif res.status_code != 200:
                print(res.content)
            elif 'mp3_key' in res.content.decode() and 'htm_key' in res.content.decode():
                nb_success += 1

        nb_reqs_total += len(reqs)
        reqs = []

        print('%d reqs total, %d reqs successful, %.3lf success rate' %
              (nb_reqs_total, nb_success, nb_success / nb_reqs_total))
        print('Downloaded %d / %d' % (len(s3_ids) + nb_success, len(S)))
        print('S index %d / %d' % (i, len(S)))
        rem = len(S) - len(s3_ids) - nb_success
        rate = nb_reqs_total / sum(req_times)
        print('Averaging %.2lf reqs / second' % rate)
        print('ETA %.3lf hours' % (rem / rate / 60 / 60))
        print('*' * 30)
