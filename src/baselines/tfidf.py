import json
import time
from pathlib import Path

import nltk
import pickledb

nltk.download("omw-1.4", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.baselines import utils


class CustomTokenizer:
    def __init__(self, stop=True, lemma=True):
        self.wnl = WordNetLemmatizer()
        self.stop = stop
        self.lemma = lemma

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            parts = k.split("__")
            if parts[0].startswith("pipeline"):
                pipe_num = int(parts[0].split("_")[1])
                param_name = "__".join(parts[1:])
                self.pipelines[pipe_num].set_params(*{param_name: v})

    def __call__(self, doc):
        if self.lemma and self.stop:
            return [
                self.wnl.lemmatize(t)
                for t in word_tokenize(doc)
                if t not in stopwords.words("english")
            ]
        elif self.lemma:
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        elif self.stop:
            return [
                t for t in word_tokenize(doc) if t not in stopwords.words("english")
            ]
        else:
            return [t for t in word_tokenize(doc)]


# replicate settings defined by Bozarth et al
# settings are 100 in doc frequency but doesn't too well with smaller corpuses
MIN_DOCUMENT_FREQUENCY = (
    1  # minimum number of users that must use word for it to be counted
)
MAX_DOCUMENT_FREQUENCY = (
    0.25  # maximum number of users that can use word for it to be counted
)


# https://github.com/lbozarth/keywordexpansion/blob/main/src_pipelines/run_tfidf.py
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def get_top_keywords(
    tweets, budget, positive_labelled_keywords, negative_labelled_keywords
):
    if len(tweets) <= 10:
        return {}

    # Bozarth et al. uses 300 max features but seems like a weird setting
    vectorizer = TfidfVectorizer(
        # tokenizer=CustomTokenizer(stop=True, lemma=True),
        ngram_range=(1, 1),
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_df=MAX_DOCUMENT_FREQUENCY,
        use_idf=True,
        # max_features=300,
    )

    results = vectorizer.fit_transform(tweets)
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = {
        feature_names[idx]: round(score, 3)
        for idx, score in sort_coo(results.tocoo())
        if feature_names[idx] not in positive_labelled_keywords
        and feature_names[idx] not in negative_labelled_keywords
    }

    # build dict matching keywords to their tf-idf scores
    return utils.kget(top_keywords, budget)


if __name__ == "__main__":
    db_filepath = Path(".tmp.db/tweets_1.db")
    tweet_db = pickledb.load(db_filepath, False)
    tweets = [tweet_db.get(key) for key in tweet_db.getall()]
    print(f"Loaded {len(tweets)} tweets for test")
    start = time.time()
    top_keywords = get_top_keywords(tweets, 30, set(), set())
    print(
        f"Found {len(top_keywords)} top keywords to use in {time.time()-start:.2f} seconds"
    )
    print(json.dumps(top_keywords, indent=4))
