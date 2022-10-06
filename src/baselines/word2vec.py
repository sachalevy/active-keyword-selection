import json
import re
import subprocess
import time
from pathlib import Path


import fasttext
import pickledb
from gensim.models import FastText

from src.baselines import utils

# implementation guided by https://github.com/lbozarth/keywordexpansion/blob/main/src_pipelines/run_cosine_similarity.py
TRAIN_FILEPATH = ".tmp.train_w2v.txt"
MODEL_FILEPATH = ".tmp.model_w2v"


def setup_model(tweets):
    if len(tweets) == 0:
        raise RuntimeError("No tweets provided")

    # write tweets to file for processing
    with open(TRAIN_FILEPATH, "w") as f:
        f.write("\n".join(tweets))

    # preprocess model using command line, output to same file
    subprocess.run(
        [
            "cat",
            TRAIN_FILEPATH,
            "|",
            "sed",
            "-e",
            "s/\([.\!?,'/()]\)/ \1 /g",
            "|",
            "tr",
            "[:upper:]",
            "[:lower:]",
            ">",
            TRAIN_FILEPATH,
        ],
        capture_output=True,
    )
    w2v_model = fasttext.train_unsupervised(
        TRAIN_FILEPATH, model="skipgram", minCount=1, dim=300, loss="hs"
    )
    w2v_model.save_model(MODEL_FILEPATH)


def process_tweets(tweets):
    # store as list of lists of words
    processed_tweets = []
    for tweet in tweets:
        tweet = re.sub(r"\([^)]*\)", "", tweet)
        tokens = re.sub(r"[^a-z0-9]+", " ", tweet.lower()).split()
        processed_tweets.append(tokens)

    return processed_tweets


def get_top_keywords(
    tweets, budget, positive_labelled_keywords, negative_labelled_keywords
):
    if len(tweets) <= 10:
        return {}

    # https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c

    # setup_model(tweets)
    # fasttext_model = load_facebook_model(MODEL_FILEPATH)

    tweets = process_tweets(tweets)
    fasttext_model = FastText(
        tweets, vector_size=300, min_count=1, workers=4, hs=1, sg=1
    )

    top_keywords = {}
    iter_count = 0
    # retrieve list of most similar to positive & dissimilar from negative keywords
    while len(top_keywords) < budget and iter_count < 4:
        top_keywords.update(
            {
                r[0]: round(r[1], 3)
                for r in fasttext_model.wv.most_similar(
                    positive=list(positive_labelled_keywords),
                    # negative=list(negative_labelled_keywords),
                    topn=budget * (iter_count + 1),
                )
                if r[0] not in positive_labelled_keywords
                and r[0] not in negative_labelled_keywords
            }
        )

        # if need to increase the number of keywords to be included
        iter_count += 1

    # remove keywords that have already been retrieved
    return utils.kget(top_keywords, budget)


if __name__ == "__main__":
    # looking at first day of data for testing purposes
    db_filepath = Path(".tmp.db/tweets_1.db")
    tweet_db = pickledb.load(db_filepath, False)
    tweets = [tweet_db.get(key) for key in tweet_db.getall()]
    print(f"Loaded {len(tweets)} tweets for test")

    start = time.time()

    with open("keywords.json", "r") as f:
        keywords = json.load(f)

    top_keywords = get_top_keywords(tweets, 30, keywords.get("LOCKDOWN"), set())
    print(
        f"Found {len(top_keywords)} top keywords to use in {time.time()-start:.2f} seconds"
    )
    print(json.dumps(top_keywords, indent=4))
