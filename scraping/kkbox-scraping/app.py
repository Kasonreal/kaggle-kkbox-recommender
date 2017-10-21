from chalice import Chalice
from chalice import BadRequestError
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import boto3
import requests

app = Chalice(app_name='kkbox-scraper')
app.debug = True


@app.route('/')
def index():
    ip = urlopen('http://ident.me').read().decode('utf8')
    return ip

# Levenshtein functions from: https://github.com/obulkin/string-dist/blob/master/stringdist/pystringdist/levenshtein.py


def levenshtein_norm(source, target):
    """Calculates the normalized Levenshtein distance between two string
    arguments. The result will be a float in the range [0.0, 1.0], with 1.0
    signifying the biggest possible distance between strings with these lengths
    """

    # Compute Levenshtein distance using helper function. The max is always
    # just the length of the longer string, so this is used to normalize result
    # before returning it
    distance = _levenshtein_compute(source, target, False)
    return float(distance) / max(len(source), len(target))


def _levenshtein_compute(source, target, rd_flag):
    """Computes the Levenshtein
    (https://en.wikipedia.org/wiki/Levenshtein_distance)
    and restricted Damerau-Levenshtein
    (https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
    distances between two Unicode strings with given lengths using the
    Wagner-Fischer algorithm
    (https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm).
    These distances are defined recursively, since the distance between two
    strings is just the cost of adjusting the last one or two characters plus
    the distance between the prefixes that exclude these characters (e.g. the
    distance between "tester" and "tested" is 1 + the distance between "teste"
    and "teste"). The Wagner-Fischer algorithm retains this idea but eliminates
    redundant computations by storing the distances between various prefixes in
    a matrix that is filled in iteratively.
    """

    # Create matrix of correct size (this is s_len + 1 * t_len + 1 so that the
    # empty prefixes "" can also be included). The leftmost column represents
    # transforming various source prefixes into an empty string, which can
    # always be done by deleting all characters in the respective prefix, and
    # the top row represents transforming the empty string into various target
    # prefixes, which can always be done by inserting every character in the
    # respective prefix. The ternary used to build the list should ensure that
    # this row and column are now filled correctly
    s_range = range(len(source) + 1)
    t_range = range(len(target) + 1)
    matrix = [[(i if j == 0 else j) for j in t_range] for i in s_range]

    # Iterate through rest of matrix, filling it in with Levenshtein
    # distances for the remaining prefix combinations
    for i in s_range[1:]:
        for j in t_range[1:]:
            # Applies the recursive logic outlined above using the values
            # stored in the matrix so far. The options for the last pair of
            # characters are deletion, insertion, and substitution, which
            # amount to dropping the source character, the target character,
            # or both and then calculating the distance for the resulting
            # prefix combo. If the characters at this point are the same, the
            # situation can be thought of as a free substitution
            del_dist = matrix[i - 1][j] + 1
            ins_dist = matrix[i][j - 1] + 1
            sub_trans_cost = 0 if source[i - 1] == target[j - 1] else 1
            sub_dist = matrix[i - 1][j - 1] + sub_trans_cost

            # Choose option that produces smallest distance
            matrix[i][j] = min(del_dist, ins_dist, sub_dist)

            # If restricted Damerau-Levenshtein was requested via the flag,
            # then there may be a fourth option: transposing the current and
            # previous characters in the source string. This can be thought of
            # as a double substitution and has a similar free case, where the
            # current and preceeding character in both strings is the same
            if rd_flag and i > 1 and j > 1 and source[i - 1] == target[j - 2] \
                    and source[i - 2] == target[j - 1]:
                trans_dist = matrix[i - 2][j - 2] + sub_trans_cost
                matrix[i][j] = min(matrix[i][j], trans_dist)

    # At this point, the matrix is full, and the biggest prefixes are just the
    # strings themselves, so this is the desired distance
    return matrix[len(source)][len(target)]


@app.route('/scrape', methods=['POST'])
def scrape():

    # Deconstruct query.
    body = app.current_request.json_body
    song_id = body['song_id']
    song_name = body['song_name']
    artist_name = body['artist_name']
    song_seconds = body['song_seconds']

    # S3 connection.
    S3 = boto3.client('s3', region_name='us-east-1')
    BUCKET = 'kkbox-scraping'

    # Paths for saving HTML page and MP3.
    htm_key = '%s_page.html' % song_id
    mp3_key = '%s_sample.mp3' % song_id

    # Retrieve search page.
    search_base = 'https://www.kkbox.com/sg/en/search.php?word='
    search_query = '%s %s' % (song_name, artist_name)
    search_url = search_base + search_query
    res = requests.get(search_url)

    # Parse the HTML page to extract titles, times, artists, links.
    htm = soup(res.content, "html.parser")
    song_items = htm.select('table.song-table tr.song-item')

    # Find the closest song from all results based on song name, artist name, and song time.
    song_url = None
    min_diff = -1
    for i, item in enumerate(song_items):
        song_name_ = item.select('a.song-title')[0].text.strip()
        song_url_ = item.select('a.song-title')[0]['href']
        artist_name_ = item.select('div.song-artist-album > a:nth-of-type(1)')[0].text.strip()
        song_seconds_ = item.select('td.song-time')[0].text.strip()
        song_seconds_ = 60 * int(song_seconds_.split(':')[0]) + int(song_seconds_.split(':')[1])

        diff = levenshtein_norm(song_name, song_name_)
        diff += levenshtein_norm(artist_name, artist_name_)
        diff += abs(song_seconds - song_seconds_) / song_seconds

        print(i, song_name_, artist_name_)
        print(levenshtein_norm(song_name, song_name_))
        print(levenshtein_norm(artist_name, artist_name_))
        print(song_seconds, song_seconds_)
        print(diff)
        print('*' * 30)

        if min_diff == -1 or diff < min_diff:
            min_diff = diff
            song_url = song_url_

    # Retreive and save the song page.
    res = requests.get('https://www.kkbox.com%s' % song_url)
    S3.put_object(Bucket=BUCKET, Key=htm_key, Body=res.content.decode())

    # Download the mp3 and save it.
    htm = soup(res.content, "html.parser")
    mp3_meta = htm.find('meta', property='music:preview_url:url')
    mp3_url = mp3_meta['content']
    S3.put_object(Bucket=BUCKET, Key=mp3_key, Body=urlopen(mp3_url).read())

    return {'htm_key': htm_key, 'mp3_key': mp3_key}
