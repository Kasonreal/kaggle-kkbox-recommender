from more_itertools import flatten
from itertools import product
from pprint import pprint

keys2feats = {
    'user-1': [('user-id', 1), ('user-gender', 'male'), ('user-age', 20)],
    'user-2': [('user-id', 2), ('user-gender', 'female'), ('user-age', 30)],
    'user-3': [('user-id', 3), ('user-gender', 'male'), ('user-age', 15)],

    'song-1': [('song-id', 1), ('song-genre', 10), ('song-genre', 20), ('song-artist', 30)],
    'song-2': [('song-id', 2), ('song-genre', 11), ('song-artist', 40)],
    'song-3': [('song-id', 3), ('song-genre', 12), ('song-artist', 50)]
}

sample_keys = [
    ['user-1', 'song-1'],
    ['user-2', 'song-2'],
    ['user-3', 'song-3']
]

targets = [0, 1, 1]

sample_interactions = []

for (k1, k2), target in zip(sample_keys, targets):
    prods = product(keys2feats[k1], keys2feats[k2])
    sample_interactions += [list(p) + [target] for p in prods]

for i in range(len(sample_interactions)):
    print(i, sample_interactions[i])

# 20 [('user-age', 30), ('song-artist', 40), 1]
# 21 [('user-id', 3), ('song-id', 3), 1]
# 22 [('user-id', 3), ('song-genre', 12), 1]
# 23 [('user-id', 3), ('song-artist', 50), 1]
# 24 [('user-gender', 'male'), ('song-id', 3), 1]
# 25 [('user-gender', 'male'), ('song-genre', 12), 1]
# 26 [('user-gender', 'male'), ('song-artist', 50), 1]
# 27 [('user-age', 15), ('song-id', 3), 1]
# 28 [('user-age', 15), ('song-genre', 12), 1]
# 29 [('user-age', 15), ('song-artist', 50), 1]
