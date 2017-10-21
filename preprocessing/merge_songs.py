import pandas as pd

if __name__ == "__main__":
    S = pd.read_csv('data/songs.csv')
    E = pd.read_csv('data/song_extra_info.csv')
    M = pd.merge(S, E, on='song_id')
    print(S.columns)
    print(E.columns)
    print(M.columns)
    print(M.head(5))
    M.to_csv('data/songs_merged.csv')
