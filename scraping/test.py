from hashlib import sha256
import requests
import json
import pdb
from sys import argv

url = 'http://localhost:8000/scrape'

if len(argv) > 1:
    url = argv[-1] + 'scrape'

body = {
    "song_id": sha256("CXoTN1eb7AI+DntdU1vbcwGRV4SCIDxZu+YD8JP8r4E=".encode()).hexdigest(),
    "song_name": "焚情",
    "artist_name": "張 信哲 (Jeff Chang)",
    "song_seconds": int(247640 / 1000)
}
res = requests.post(url, json=body)
print(res.content)
