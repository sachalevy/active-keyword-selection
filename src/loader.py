import itertools
import json
import pickle
import random
import time
from pathlib import Path

random.seed(123)

from tqdm import tqdm

from src.utils import transaction, dynamic_loader

LOCAL_ECHEN_FILEPATH = Path("data/echen_hashtag_relations.json")
SAMPLE_ECHEN_FILEPATH = Path("data/echen_hashtag_relations_sample.json")
DEFAULT_DUMP_FILEPATH = Path(".tmp.graph_data_loader.pkl")
DEFAULT_USERS_DUMP_FILEPATH = Path(".tmp.graph_users_data_loader.pkl")
DEFAULT_STATIC_DUMP_FILEPATH = Path(".tmp.static_graph_data_loader.pkl")
LOCAL_KEYWORDS_FILEPATH = Path("data/keywords.json")
LIGHT_LOADER_DUMP_FILEPATH = Path(".tmp.light_graph_data_loader.pkl")
DEFAULT_EXPERIMENT_TOPICS = ["MASK", "LOCKDOWN", "VACCINE"]
DEFAULT_KEYWORD_SAMPLING_METHODS = ["random", "degree"]
DEFAULT_KEYWORD_SAMPLING_COUNTS = [10]
DEFAULT_BUDGETS = [3, 10, 30]


class TemporalExperimentDataLoader:
    def __init__(
        self,
        data_filepath: str = LOCAL_ECHEN_FILEPATH,
        expected_relation_count: int = int(28400000),
        keywords_filepath: str = LOCAL_KEYWORDS_FILEPATH,
        day_count: int = 31,
        use_users: bool = False,
        stop: int = None,
        experiment_topics: list = DEFAULT_EXPERIMENT_TOPICS,
        keyword_sampling_methods: list = DEFAULT_KEYWORD_SAMPLING_METHODS,
        keyword_sampling_counts: list = DEFAULT_KEYWORD_SAMPLING_COUNTS,
        evaluation_budgets: list = DEFAULT_BUDGETS,
    ):
        self.data_filepath = data_filepath
        self.use_users = use_users
        self.expected_relation_count = expected_relation_count
        self.keywords_filepath = keywords_filepath
        self.day_count = day_count
        self.evaluation_budgets = evaluation_budgets

        self.window_size = 3
        self.iter_days = 0
        self.results_file = (
            "results_user_graph.json" if self.use_users else "results_tweet_graph.json"
        )

        self.experiment_topics = experiment_topics
        self.keyword_sampling_methods = keyword_sampling_methods
        self.keyword_sampling_counts = keyword_sampling_counts
        self.negative_key = "negative_keywords"
        self.positive_key = "positive_keywords"
        self.baseline_key = "baseline_keywords"
        self.oracle_key = "oracle_keywords"

        # generate data configuration for experiments
        self.param_matrix = [
            list(range(1, self.day_count)),
            self.experiment_topics,
            self.keyword_sampling_counts,
            self.evaluation_budgets,
            self.keyword_sampling_methods,
        ]
        self.configs = transaction.get_keys(*tuple(self.param_matrix[1:]))
        self.day_configs = transaction.get_keys(*(self.param_matrix))
        print(
            f"Running experiments on {len(self.day_configs)} independent configurations"
        )
        self.hashtag_count, self.tweet_count, self.user_count = 0, 0, 0
        self.hashtag_idx, self.idx_hashtag = {}, {}
        self.tweet_idx, self.idx_tweet = {}, {}
        self.oracle_keywords, self.search_oracle_keywords = {}, {}
        self.oracle_tweets, self.oracle_users = {}, {}
        self.user_idx, self.idx_user = {}, {}
        self.tweet_user_dict, self.user_tweet_dict = {}, {}
        self.total_hashtag_occurences = {}
        self.daily_hashtag_occurences = {i: {} for i in range(1, self.day_count)}
        self.current_graph_batch = {topic: {} for topic in self.experiment_topics}

        # hashtag graph specific attributes
        self.dump_filepath = (
            DEFAULT_DUMP_FILEPATH if not self.use_users else DEFAULT_USERS_DUMP_FILEPATH
        )

        self.load_preliminary_keywords()
        self.load_data()
        self.keywords = self.load_experiment_keywords()
        print(
            "Here are the daily counts for retrieved hashtags",
            [
                sum(
                    [
                        self.daily_hashtag_occurences[i][h]
                        for h in self.daily_hashtag_occurences[i]
                    ]
                )
                for i in self.daily_hashtag_occurences
            ],
        )

    def load_preliminary_keywords(self):
        with open(str(self.keywords_filepath), "r") as fd:
            self.oracle_keywords = json.load(fd)

        for topic in self.oracle_keywords:
            for keyword in self.oracle_keywords[topic]:
                self.search_oracle_keywords[keyword] = topic

            self.oracle_tweets[topic] = set()
            self.oracle_users[topic] = set()

    def load_experiment_keywords(self) -> dict:
        keywords = transaction.rinit(self.day_configs)
        self.oracle_keywords = {}
        with open(str(self.keywords_filepath), "r") as fd:
            stored_keywords = json.load(fd)

        for topic in self.experiment_topics:
            # make sure always have enough keywords to be prompting
            tmp_keywords = self.find_always_available_keywords(
                set(stored_keywords.get(topic))
            )
            if len(stored_keywords.get(topic)) <= 20 or len(tmp_keywords) <= 20:
                oracle_keywords = set(stored_keywords.get(topic))
            else:
                oracle_keywords = set(tmp_keywords)

            self.oracle_keywords[topic] = oracle_keywords
            print(
                f"Loading oracle keywords for {topic}: {100*float(len(oracle_keywords)/len(stored_keywords.get(topic))):.2f}% (available {len(oracle_keywords)}/total {len(stored_keywords.get(topic))})"
            )
            for idx, (day, count, budget, method) in enumerate(
                transaction.get_keys(
                    *tuple([self.param_matrix[0]] + self.param_matrix[2:])
                )
            ):
                negative_keypath = (
                    day,
                    topic,
                    count,
                    budget,
                    method,
                    self.negative_key,
                )
                transaction.kset(keywords, set(), *negative_keypath)
                oracle_keypath = negative_keypath[:-1] + (self.oracle_key,)
                transaction.kset(keywords, oracle_keywords, *oracle_keypath)
                baseline_keypath = negative_keypath[:-1] + (self.baseline_key,)
                baseline_keywords = self.get_baseline_keywords(
                    oracle_keywords,
                    sampling_count=count,
                    sampling_method="degree",
                )
                transaction.kset(keywords, baseline_keywords, *baseline_keypath)

        return keywords

    def find_available_keywords(self, keywords: set) -> set:
        """Retrieve keywords appearing at least once in the dataset."""
        return set(self.hashtag_idx).intersection(keywords)

    def find_always_available_keywords(self, stored_keywords: set) -> set:
        """Retain only keywords available at least once every day during
        the experiment.

        :param stored_keywords: keywords to check on
        :type stored_keywords: set
        :return: filtered set of keywords
        :rtype: set
        """
        to_remove = set()
        for day in self.daily_hashtag_occurences:
            # get all keywords available on that day
            available_keywords = set(
                self.daily_hashtag_occurences.get(day)
            ).intersection(stored_keywords)
            # retrieve those that weren't available
            to_remove.update(set(stored_keywords).difference(available_keywords))

        return set(stored_keywords) - to_remove

    def get_baseline_keywords(
        self,
        oracle_keywords: set,
        sampling_count: int = 10,
        sampling_method: str = "degree",
    ) -> set:
        baseline_keywords = set()
        if sampling_method == "degree":
            baseline_keywords = self.get_baseline_keywords_degree(
                oracle_keywords, sampling_count
            )
        elif sampling_method == "random":
            baseline_keywords = self.get_baseline_keywords_random(
                oracle_keywords, sampling_count
            )
        else:
            raise ValueError("Unknown keyword sampling method")

        return baseline_keywords

    def get_baseline_keywords_degree(
        self, oracle_keywords: set, sampling_count: int = 10
    ) -> set:
        sorted_hashtags = self.get_sorted_oracle_keywords_on_day(oracle_keywords, day=1)
        if not sorted_hashtags or len(sorted_hashtags) < sampling_count:
            # raise ValueError("Hashtag graph not loaded")
            sorted_hashtags = list(oracle_keywords)

        return set(sorted_hashtags[:sampling_count])

    def get_sorted_oracle_keywords_on_day(
        self, oracle_keywords: set, day: int = 1
    ) -> list:
        if len(self.daily_hashtag_occurences.get(day)) == 0:
            return None

        # sort first by hashtag count, then by length of the keywords
        # can justify this choise by optimizing the query length for Twitter API
        return [
            k
            for k, _ in sorted(
                {
                    keyword: self.daily_hashtag_occurences.get(day).get(keyword)
                    for keyword in set(
                        self.daily_hashtag_occurences.get(day)
                    ).intersection(oracle_keywords)
                }.items(),
                key=lambda item: (item[1], -len(item[0])),
                reverse=True,
            )
        ]

    def get_oracle_user_coverage(self, test_set, topic):
        return len(test_set.intersection(self.oracle_users.get(topic))) / len(
            self.oracle_users.get(topic)
        )

    def get_oracle_tweet_coverage(self, test_set, topic):
        return len(test_set.intersection(self.oracle_tweets.get(topic))) / len(
            self.oracle_tweets.get(topic)
        )

    def get_baseline_keywords_random(
        self, oracle_keywords: set, sampling_count: int = 10
    ) -> set:
        return set(random.sample(list(oracle_keywords), sampling_count))

    def get_day_idx(self, x):
        date = x.get("created_at").get("$date").split("T")[0].split("-")
        return int(date[-1]), int(date[-2])

    def get_user_set(self, tweet_set):
        return set(
            itertools.chain(
                *[
                    list(self.tweet_user_dict.get(self.tweet_idx.get(t)))
                    for t in tweet_set
                ]
            )
        )

    def get_current_tweet_set(self, source_hashtag_dict, day=None):
        if not self.use_users:
            return set(source_hashtag_dict)
        else:
            # assuming that most recently loaded batch yielded increment in day count
            day = self.iter_days if not day else day
            if self.user_tweet_dict.get(day):
                tweet_set = set(
                    itertools.chain(
                        *[
                            list(
                                self.user_tweet_dict.get(day).get(self.user_idx.get(u))
                            )
                            for u in source_hashtag_dict
                        ]
                    )
                )
                return {self.idx_tweet.get(t) for t in tweet_set}
            else:
                return set()

    def get_current_user_set(self, source_hashtag_dict):
        if self.use_users:
            return set(source_hashtag_dict)
        else:
            user_set = set(
                itertools.chain(
                    *[
                        list(self.tweet_user_dict.get(self.tweet_idx.get(t)))
                        for t in source_hashtag_dict
                    ]
                )
            )
            return {self.idx_user.get(u) for u in user_set}

    def load_data(self):
        self.day_batches = {i: set() for i in range(1, self.day_count)}
        processed_relations = 0

        with open(self.data_filepath, "r") as fd:
            iterator = dynamic_loader.MongoExportIterator(fd)
            for hashtag_relation in (
                iterator
                if not self.expected_relation_count
                else tqdm(iterator, total=self.expected_relation_count)
            ):
                # skip 31st of march & only 30 days in April
                day, month = self.get_day_idx(hashtag_relation)
                if day >= self.day_count or (
                    month == 5 and day == 1
                ):  # day count for april is 31
                    continue

                hashtag = hashtag_relation["hashtag"].lower().replace("ー", "-")
                if hashtag not in self.hashtag_idx:
                    self.hashtag_idx[hashtag] = self.hashtag_count
                    self.idx_hashtag[self.hashtag_count] = hashtag
                    self.hashtag_count += 1

                if hashtag_relation.get("tid") not in self.tweet_idx:
                    self.tweet_idx[hashtag_relation.get("tid")] = self.tweet_count
                    self.idx_tweet[self.tweet_count] = hashtag_relation.get("tid")
                    self.tweet_count += 1

                if day not in self.user_tweet_dict:
                    self.user_tweet_dict[day] = {}

                if hashtag_relation.get("user_id") not in self.user_idx:
                    self.user_idx[hashtag_relation.get("user_id")] = self.user_count
                    self.idx_user[self.user_count] = hashtag_relation.get("user_id")
                    self.user_count += 1

                if hashtag in self.search_oracle_keywords:
                    self.oracle_tweets[self.search_oracle_keywords[hashtag]].add(
                        self.tweet_idx[hashtag_relation.get("tid")]
                    )
                    self.oracle_users[self.search_oracle_keywords[hashtag]].add(
                        self.user_idx[hashtag_relation.get("user_id")]
                    )

                if (
                    self.tweet_idx[hashtag_relation.get("tid")]
                    not in self.tweet_user_dict
                ):
                    self.tweet_user_dict[
                        self.tweet_idx[hashtag_relation.get("tid")]
                    ] = set()
                self.tweet_user_dict[self.tweet_idx[hashtag_relation.get("tid")]].add(
                    self.user_idx[hashtag_relation.get("user_id")]
                )

                # build the reverse user-tweet dict compartimentalized by days
                if (
                    self.user_idx[hashtag_relation.get("user_id")]
                    not in self.user_tweet_dict[day]
                ):
                    self.user_tweet_dict[day][
                        self.user_idx.get(hashtag_relation.get("user_id"))
                    ] = set()
                self.user_tweet_dict[day][
                    self.user_idx.get(hashtag_relation.get("user_id"))
                ].add(self.tweet_idx[hashtag_relation.get("tid")])

                if hashtag not in self.daily_hashtag_occurences[day]:
                    self.daily_hashtag_occurences[day][hashtag] = 0
                self.daily_hashtag_occurences[day][hashtag] += 1

                # build edge list taking into account which type of graph being built
                if not self.use_users:
                    vertex = self.tweet_idx.get(hashtag_relation.get("tid"))
                else:
                    vertex = self.user_idx.get(hashtag_relation.get("user_id"))

                # add a new edge with idx of hashtag & source
                self.day_batches[day].add((self.hashtag_idx.get(hashtag), vertex))
                processed_relations += 1

        print(f"Processed {processed_relations} hashtag relations.")
        print(
            f"Hashtag count: {self.hashtag_count}, tweet count: {self.tweet_count}, user count: {self.user_count}."
        )

    def __iter__(
        self,
    ):
        self.iter_days = 0
        return self

    def get_evicted_keywords(self) -> dict:
        """Remove hashtags more popular than most baseline keyword with highest degree."""
        evicted_keywords = transaction.rinit(self.configs)
        sorted_occurences = sorted(
            self.daily_hashtag_occurences.get(self.iter_days).items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for config in map(lambda x: transaction.ParamConfig(*x), self.configs):
            transaction.kset(evicted_keywords, set(), *config)
            baseline_keypath = (self.iter_days, *config, self.baseline_key)
            baseline_keywords = transaction.rget(self.keywords, *baseline_keypath)
            evicted_keywords_config = transaction.rget(evicted_keywords, *config)
            for hashtag, _ in sorted_occurences:
                if hashtag in baseline_keywords:
                    break
                else:
                    evicted_keywords_config.add(hashtag)

            # print(
            #    "For experiment config",
            #    *config,
            #    "evicted",
            #    len(evicted_keywords_config),
            #    "keywords from graph dataset",
            #    evicted_keywords_config,
            # )

        return evicted_keywords

    def __next__(self):
        self.iter_days += 1
        if self.iter_days >= self.day_count:  # no 31st of April in experiments
            raise StopIteration()

        start = time.time()
        evicted_keywords = self.get_evicted_keywords()
        # employ a proxy variabel to switch between users & tweets based graphs
        source_idx_handle = self.user_idx if self.use_users else self.tweet_idx
        idx_source_handle = self.idx_user if self.use_users else self.idx_tweet

        day = 1 if self.iter_days == 1 else self.iter_days - 1

        self.current_graph_batch = transaction.rinit(self.day_configs)
        for idx, config in enumerate(
            map(lambda x: transaction.ParamConfig(*x), self.configs)
        ):
            th_dict, ht_dict = dict(), dict()
            transaction.kset(
                self.current_graph_batch,
                th_dict,
                *(self.iter_days, *config, transaction.th_key),
            )
            transaction.kset(
                self.current_graph_batch,
                ht_dict,
                *(self.iter_days, *config, transaction.ht_key),
            )

            # remove evicted_keywords
            current_evicted_keywords = transaction.rget(evicted_keywords, *config)
            baseline_keypath = (day, *config, self.baseline_key)
            current_baseline_keywords = transaction.rget(
                self.keywords, *baseline_keypath
            )

            direct_edges = [
                edge
                for edge in self.day_batches[self.iter_days]
                if (
                    (self.idx_hashtag[edge[0]] in current_baseline_keywords)
                    and (self.idx_hashtag[edge[0]] not in current_evicted_keywords)
                )
            ]

            for edge in direct_edges:
                if self.idx_hashtag[edge[0]] not in ht_dict:
                    ht_dict[self.idx_hashtag[edge[0]]] = set()
                ht_dict[self.idx_hashtag[edge[0]]].add(idx_source_handle[edge[1]])

                if idx_source_handle[edge[1]] not in th_dict:
                    th_dict[idx_source_handle[edge[1]]] = list()
                th_dict[idx_source_handle[edge[1]]].append(self.idx_hashtag[edge[0]])

            extra_edges = [
                edge
                for edge in self.day_batches[self.iter_days]
                if idx_source_handle[edge[1]] in th_dict
                and self.idx_hashtag[edge[0]] not in current_baseline_keywords
                and self.idx_hashtag[edge[0]] not in current_evicted_keywords
            ]

            # print(f"Day {self.iter_days} have {len(extra_edges)} 1hop edges")
            for edge in extra_edges:
                if self.idx_hashtag[edge[0]] not in ht_dict:
                    ht_dict[self.idx_hashtag[edge[0]]] = set()
                ht_dict[self.idx_hashtag[edge[0]]].add(idx_source_handle[edge[1]])

                if idx_source_handle[edge[1]] not in th_dict:
                    # tweet can quote many hashtags
                    th_dict[idx_source_handle[edge[1]]] = list()
                th_dict[idx_source_handle[edge[1]]].append(self.idx_hashtag[edge[0]])

    def dump(self):
        with open(str(self.dump_filepath), "wb") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls, use_users=False):
        dump_filepath = (
            DEFAULT_DUMP_FILEPATH if not use_users else DEFAULT_USERS_DUMP_FILEPATH
        )
        if dump_filepath.is_file():
            with open(str(dump_filepath), "rb") as fd:
                inst = pickle.load(fd)
        else:
            raise RuntimeError("cannot find dump file")
        return inst
