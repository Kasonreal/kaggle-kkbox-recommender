# kaggle-kkbox-recommender

Models and code for the [Kaggle KKBox Music Recommendation Competition](kaggle.com/c/kkbox-music-recommendation-challenge). I ended up in 222nd place. The code is messy, but may end up useful for someone.

I spent a lot of time trying various collaborative filtering and factorization-based models. Gradient boosted decision trees ended up working much better. I also briefly explored a model that used a convolutional network to learn song vectors from spectrograms of the song audio. This required scraping/downloading the 30-second sample MP3s from KKBox; I did this using AWS Lambda and S3. My best submission was an equally-weighted ensemble of GBDT and Factorization Machines.
