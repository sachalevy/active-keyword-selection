import copy
import html
import json
import pickle
import re
import sys
import unicodedata
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import emoji
import pickledb
import pymongo
import unidecode
from tqdm import tqdm

from src.utils.dynamic_loader import MongoExportIterator

DEFAULT_DUMP_FILEPATH = Path(".tmp.tweet_loader_dump.pkl")
TWEET_DATASET_FILEPATH = Path("/Users/sachalevy/Downloads/echen_fullv1_tweets.json")
FULL_TWEET_DATASET_FILEPATH = Path("data/echen_tweets/")

# initialize db dir
DB_ROOT_DIR = Path(".tmp.db")
if not DB_ROOT_DIR.is_dir():
    DB_ROOT_DIR.mkdir(parents=True)

# compile regexes
# username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))")
control_char_regex = re.compile(r"[\r\n\t]+")
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip("''´" "–-", "'''\"\"--")])
# HTML parser
html_parser = HTMLParser()


def preprocess_bert(text):
    """Preprocesses input for BERT"""
    # standardize
    text = standardize_text(text)
    text = asciify_emojis(text)
    text = standardize_punctuation(text)
    text = text.lower()
    text = remove_unicode_symbols(text)
    text = remove_accented_characters(text)
    return text


def remove_accented_characters(text):
    text = unidecode.unidecode(text)
    return text


def remove_unicode_symbols(text):
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "So")
    return text


def asciify_emojis(text):
    """
    Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
    For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text


def standardize_text(text):
    """
    1) Escape HTML
    2) Replaces some non-standard punctuation with standard versions.
    3) Replace \r, \n and \t with white spaces
    4) Removes all other control characters and the NULL byte
    5) Removes duplicate white spaces
    """
    # escape HTML symbols
    text = html.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace("…", "...")
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, " ", text)
    # remove all remaining control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # replace multiple spaces with single space
    text = " ".join(text.split())
    return text.strip()


def standardize_punctuation(text):
    return "".join(
        [
            unidecode.unidecode(t) if unicodedata.category(t)[0] == "P" else t
            for t in text
        ]
    )


get_day = lambda x: int(
    x.get("json").get("created_at").get("$date").split("T")[0].split("-")[-1]
)


def download_echen_data_partition(start_date, end_date, batch_idx):
    """Retrieve tweets from a MongoDB and pre-process their text.

    :param start_date: Date to start downloading tweets from.
    :type start_date: datetime
    :param end_date: Date to end downloading tweets from.
    :type end_date: datetime
    :param batch_idx: Id of the time partition being downloaded.
    :type batch_idx: int
    """
    from snpipeline.db import db_client

    db_conn = db_client.DBClient()

    # initialize connections to databases
    if end_date.month == 5 and end_date.day == 1:
        end_day = 31
    else:
        end_day = end_date.day + 1

    tweet_dbs = {
        day_idx: pickledb.load(
            DB_ROOT_DIR.joinpath(f"tweets_{day_idx}_{batch_idx}.db"), False
        )
        for day_idx in range(start_date.day, end_day)
    }
    print(f"Loading days {start_date} to {end_date} from DB for batch {batch_idx}")

    user_tweet_dict = {}
    user_tweet_filepath = f"user_tweet_{batch_idx}.json"
    tweet_count = 0
    tweet_collection = db_conn._get_collection(
        "tweets", db_name="echen_keyword_study_v1"
    )
    projection = {
        "tid": True,
        "json.created_at": True,
        "json.full_text": True,
        "uid": True,
    }
    sort = [("json.created_at", pymongo.ASCENDING)]
    doc_filter = {
        "json.created_at": {"$gte": start_date, "$lt": end_date},
        "json.retweeted_status": {"$exists": False},
    }
    for tweet in tweet_collection.find(
        filter=doc_filter,
        sort=sort,
        projection=projection,
        batch_size=4096,
        allow_disk_use=True,
    ):
        tweet_count += 1
        processed_text = preprocess_bert(tweet["json"]["full_text"])
        tweet_dbs.get(tweet["json"]["created_at"].day).set(tweet["tid"], processed_text)

        if tweet["uid"] not in user_tweet_dict:
            user_tweet_dict[tweet["uid"]] = set()
        user_tweet_dict[tweet["uid"]].add(tweet["tid"])

        if tweet_count % 256000 == 0:
            print(
                f"on {batch_idx} retrieved {tweet_count} tweets for {len(user_tweet_dict)} users"
            )

    # save all dbs
    for db in tweet_dbs:
        tweet_dbs.get(db).dump()

    for user in user_tweet_dict:
        user_tweet_dict[user] = list(user_tweet_dict[user])
    print(f"found {len(user_tweet_dict)} users and {tweet_count} tweets")

    with open(user_tweet_filepath, "w") as f:
        json.dump(user_tweet_dict, f)

    return


def merge_tweets():
    import snpipeline

    day_count = 31
    db_files = {}
    for day_idx in range(1, day_count):
        db_files[day_idx] = []
        for batch_idx in range(snpipeline.settings.CPU_COUNT):
            if DB_ROOT_DIR.joinpath(f"tweets_{day_idx}_{batch_idx}.db").is_file():
                db_files[day_idx].append(
                    DB_ROOT_DIR.joinpath(f"tweets_{day_idx}_{batch_idx}.db")
                )
    print(f"seeing {len(db_files)} db files in fs")

    new_root_dir = Path(".tmp.tweetsdb")
    if not new_root_dir.is_dir():
        new_root_dir.mkdir(parents=True)

    for day_idx in db_files:
        new_db_file = new_root_dir.joinpath(f"tweets_{day_idx}.db")
        tmp_db = pickledb.load(new_db_file, False)

        if len(db_files[day_idx]) == 1:
            db = pickledb.load(db_files[day_idx][0], False)
            tmp_db.db = copy.deepcopy(db.db)
            print(
                f"copied from original db {len(tmp_db.db)} tweets on {db_files[day_idx][0]}"
            )
        else:
            for db_file in db_files[day_idx]:
                db = pickledb.load(db_file, False)
                tmp_db.db.update(db.db)
                print(f"found {len(db.db)} tweets in {db_file}")
        print(f"merging day {day_idx} with {len(tmp_db.db)} records in db")
        tmp_db.dump()


class FastTweetLoader(MongoExportIterator):
    def __init__(self, fd=None, day_count=31):
        super().__init__(fd)

        # store all tweets into a db and keep index in memory
        self.day_count = day_count
        self.total_tweet_count = 20800000
        self.dump_filepath = DEFAULT_DUMP_FILEPATH
        self.tweet_ids_per_day = {i: set() for i in range(1, self.day_count)}
        # initialize db dir
        self.db_root_path = Path(".tmp.db")
        if not self.db_root_path.is_dir():
            self.db_root_path.mkdir(parents=True)
        # construct db filepaths
        self.tweet_db_files = [
            self.db_root_path.joinpath(f"tweets_{i}.db")
            for i in range(1, self.day_count)
        ]
        self.tweet_dbs = {
            i: pickledb.load(self.tweet_db_files[i - 1], False)
            for i in range(1, self.day_count)
        }

    def download_tweets_from_db(self):
        import snpipeline
        from snpipeline.utils import parallel

        snpipeline.settings.load_config(".env")
        # defining time bounds to download tweets from
        start_date, end_date = datetime(year=2020, month=4, day=1), datetime(
            year=2020, month=5, day=1
        )
        delta = (end_date - start_date) / snpipeline.settings.CPU_COUNT
        time_partitions = [
            (start_date + (delta * i), start_date + (delta * (i + 1)))
            for i in range(snpipeline.settings.CPU_COUNT)
        ]

        kwargs_list = []
        for idx, (ts, te) in enumerate(time_partitions):
            kwargs_list.append({"start_date": ts, "end_date": te, "batch_idx": idx})

        print("Running download process partitioned")
        parallel.run_parallel(
            download_echen_data_partition,
            kwargs_list,
            max_workers=snpipeline.settings.CPU_COUNT,
        )

    def load_tweet_texts(self):
        """Load tweet texts from a fixed file"""
        skipped_tweet_count, processed_tweet_count = 0, 0
        for tweet in tqdm(self, total=self.total_tweet_count):
            # skip 31st of march & only 30 days in April
            day = int(
                tweet.get("json")
                .get("created_at")
                .get("$date")
                .split("T")[0]
                .split("-")[-1]
            )
            if day == self.day_count:
                continue

            # only extract tweet text (discard retweet & quotes)
            if (
                "retweeted_status" in tweet["json"]
                or "quoted_status" in tweet["json"]
                or tweet["json"]["in_reply_to_status_id_str"]
            ):
                skipped_tweet_count += 1
                continue

            self.tweet_ids_per_day[day].add(tweet["tid"])

            # process text while loading it here
            processed_text = preprocess_bert(tweet["json"]["full_text"])
            self.tweet_dbs.get(day).set(tweet["tid"], processed_text)

            processed_tweet_count += 1

            # if processed_tweet_count % 10000 == 0:
            #    break

        print("skipped {} tweets".format(skipped_tweet_count))
        print("processed {} tweets".format(processed_tweet_count))

        print("\n\nsummary of db sizes:")

        # dump all db to their backup files
        for db_idx in self.tweet_dbs:
            print(f"day {db_idx}: {len(self.tweet_ids_per_day[db_idx])} tweets")
            self.tweet_dbs.get(db_idx).dump()

    def get_tweets(self, tweet_ids: list, day: int):
        db = self.tweet_dbs.get(day)
        tweets = set()
        for tid in tweet_ids:
            tweet = db.get(tid)
            if tweet:
                # feed processed tweet to model
                t = re.sub("https:\/\/t\.co[\S]+", "URL", tweet.replace("\xa0", " "))
                t = re.sub("@[^\s]+", "USER", t)
                tweets.add(t.replace("#", " "))

        return tweets

    def dump(self):
        # remove all databases from memory
        self.tweet_dbs = None
        with open(str(self.dump_filepath), "wb") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls):
        if DEFAULT_DUMP_FILEPATH.is_file():
            with open(str(DEFAULT_DUMP_FILEPATH), "rb") as fd:
                inst = pickle.load(fd)
        else:
            raise RuntimeError("dump file not found")
        return inst


def kget(top_keywords, budget):
    return {
        k: v
        for k, v in list(
            sorted(top_keywords.items(), key=lambda item: item[1], reverse=True)
        )[:budget]
    }


def download_tweets():
    tweet_loader = FastTweetLoader()
    tweet_loader.download_tweets_from_db()


def load_tweets():
    with open(TWEET_DATASET_FILEPATH, "r") as f:
        tweet_loader = FastTweetLoader(f)
        tweet_loader.load_tweet_texts()
    tweet_loader.dump()


def profile_load_tweets(count: int = 1000):
    import cProfile
    import pstats
    from pstats import SortKey

    cProfile.run(
        """FastTweetLoader(open(TWEET_DATASET_FILEPATH, "r")).load_tweet_texts()""",
        ".tmp.profile",
    )
    p = pstats.Stats(".tmp.profile")
    print("sorted by cumulative time")
    p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    p.sort_stats(SortKey.TIME).print_stats(20)


def preprocess_tweets(tweets):
    # replace links with "URL," and non-breaking spaces (\xa0) with normal ones
    tweet_list = [tweet["text"] for tweet in tweets]
    tweet_list = [
        re.sub("https:\/\/t\.co[\S]+", "URL", tweet.replace("\xa0", " "))
        for tweet in tweet_list
    ]
    tweet_list = [tweet["json"]["text"] for tweet in tweets]
    user_list = [str(tweet["json"]["user"]["id"]) for tweet in tweets]

    # group tweets by user
    user_set = set(user_list)
    user_tweets = {key: "" for key in user_set}
    for idx, tweet in enumerate(tweet_list):
        user_tweets[user_list[idx]] = user_tweets[user_list[idx]] + " " + tweet

    # make ordered lists of tweets and user IDs to prepare for tfidf
    user_tweet_list = []
    user_id_list = []
    for key, value in user_tweets.items():
        user_tweet_list.append(value)
        user_id_list.append(key)


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download_tweets()
    elif sys.argv[1] == "merge":
        merge_tweets()
    else:
        print("Unrecognized command, try 'download' or 'merge'")
        sys.exit(1)
