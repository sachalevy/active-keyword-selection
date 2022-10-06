import copy
import json
import queue
import random
import time

from tqdm import tqdm

from src import graph, loader
from src.utils import transaction
from src.baselines import tfidf, word2vec
from src.baselines.utils import FastTweetLoader

GRAPH_BASED_SAMPLING_METHOD = ["ours", "degree", "random"]
TEXT_BASED_SAMPLING_METHOD = ["tinybert", "tfidf"]


class TemporalExperimentRunner:
    def __init__(
        self,
        data: loader.TemporalExperimentDataLoader,
        text_data: FastTweetLoader = None,
    ):
        """Wrapper to run experiments on pre-loaded dataset

        :param data: Pre-loaded dataset containing tweet-hashtag graph info.
        :type data: loader.ExperimentDataLoader
        """
        self.data = data
        self.text_data = text_data
        self.graph = {}
        self.centralities = {}
        self.labelling_actions = 0
        self.topic_queue = None
        self.runtime_recorder = {}
        self.init_results()

    def init_results(self) -> tuple:
        """Initialize the output data structure to store the results for each run"""
        self.candidate_results = transaction.rinit(self.data.day_configs)
        self.labelled_keywords_results = transaction.rinit(self.data.day_configs)
        init_paths = transaction.rinsertm(self.data.day_configs, self.data.baseline_key)
        init_vals = list(
            transaction.rget(self.data.keywords, *path) for path in init_paths
        )

        keys = [
            "seed",
            "labelled",
            "users",
            "tweets",
            "tweet_coverage",
            "user_coverage",
        ]
        vals = [init_vals, *[[set()] * len(init_vals)] * 3, *[[0] * len(init_vals)] * 2]
        for key, val in zip(keys, vals):
            transaction.rsetm(
                self.labelled_keywords_results, self.data.day_configs, key, val
            )

    def run_experiments(self):
        """Run experiments for all configurations defined in data loader."""
        self.runtime_recorder = {}
        not_skipped = 0
        _ = iter(self.data)  # setup data for iteration

        for idx, config in enumerate(
            map(lambda x: transaction.Config(*x), self.data.day_configs)
        ):
            if config.day > self.data.iter_days:
                next(self.data)

            start_time = time.time()
            prev_config = (
                config
                if config.day == 1
                else transaction.Config(*(config.day - 1, *config[1:]))
            )

            # TODO: rephrase the ht/th since switched to source-hashtag notation
            th_keypath = transaction.TweetHashtagKeyPath(*config, transaction.th_key)
            ht_keypath = transaction.HashtagTweetKeyPath(*config, transaction.ht_key)

            self.hashtag_source_dict = transaction.rget(
                self.data.current_graph_batch, *ht_keypath
            )
            self.source_hashtag_dict = transaction.rget(
                self.data.current_graph_batch, *th_keypath
            )

            # collect current hashtags for configuration
            available_hashtags = set(self.hashtag_source_dict)
            positive_keypath = (*prev_config, self.data.baseline_key)
            all_baseline_keywords = transaction.rget(
                self.data.keywords, *positive_keypath
            )
            negative_keypath = (*prev_config, self.data.negative_key)
            all_negative_keywords = transaction.rget(
                self.data.keywords, *negative_keypath
            )

            # setup proxy variables
            self.budget = config.budget

            # initialize the experiment sets for current iteration
            oracle_keypath = (*prev_config, self.data.oracle_key)
            all_oracle_keywords = transaction.rget(self.data.keywords, *oracle_keypath)
            oracle_keywords = (
                all_oracle_keywords.intersection(available_hashtags)
                - all_baseline_keywords
            )
            active_oracle_keypath = (*config, "active_oracles")
            transaction.kset(
                self.candidate_results, oracle_keywords, *active_oracle_keypath
            )

            # initialize the sets holding the keyword states pre and post labelling
            self.init_experiment_sets(
                all_baseline_keywords.intersection(available_hashtags),
                all_negative_keywords.intersection(available_hashtags),
            )

            start = time.time()
            if len(self.hashtag_source_dict) <= 20:
                self.roll_over_keyword_sets(config, prev_config)
                print(
                    f"{idx+1}/{len(self.data.day_configs)} - skipped in {time.time() - start_time} - {config}"
                )
                continue

            if config.method not in GRAPH_BASED_SAMPLING_METHOD:
                top_keywords = self.run_text_baseline(th_keypath)
                # print(
                #    f"{idx+1}/{len(self.data.day_configs)} retrieved from {config.method}",
                #    top_keywords,
                # )
                self.run_text_experiment_iterations(
                    top_keywords=top_keywords,
                    config=config,
                    oracle_keywords=oracle_keywords,
                )
            else:
                if config.method == "ours":
                    self.centralities = self.get_hashtag_scores(
                        positive_labels=self.positive_labelled_keywords,
                        negative_labels=self.negative_labelled_keywords,
                    )
                elif config.method == "random":
                    self.centralities = self.get_random_hashtag_scores(
                        positive_labels=self.positive_labelled_keywords,
                        negative_labels=self.negative_labelled_keywords,
                    )
                elif config.method == "degree":
                    self.graph = graph.generate_graph(self.hashtag_source_dict)
                    self.centralities = graph.get_centrality_measure(
                        self.graph,
                        centrality_measure=config.method,
                        positive_labels=self.positive_labelled_keywords,
                        negative_labels=self.negative_labelled_keywords,
                    )

                self.init_topic_queue()

                # go through the topic queue and label keywords
                iterations = 0
                while self.budget > 0:
                    iterations += 1
                    if not self.run_experiment_iteration(oracle_keywords):
                        break

            runtime_recorder_key = "-".join(map(str, config))
            self.runtime_recorder[runtime_recorder_key] = {
                "runtime": time.time() - start,
            }
            self.roll_over_keyword_sets(config, prev_config)

            self.graph, self.centralities = None, None
            not_skipped += 1
            print(
                f"{idx+1}/{len(self.data.day_configs)} - done in {time.time() - start_time} - {config}"
            )

        with open("runtime_recorder.json", "w") as f:
            json.dump(self.runtime_recorder, f)

        print("days not skipped", not_skipped)

    def roll_over_keyword_sets(self, config, prev_config):
        """Save all results for experiment report."""

        prev_positive_keypath = (*prev_config, self.data.baseline_key)
        prev_positive_keywords = transaction.rget(
            self.data.keywords, *prev_positive_keypath
        )
        positive_keypath = (*config, self.data.baseline_key)
        transaction.kset(
            self.data.keywords,
            copy.deepcopy(
                self.positive_labelled_keywords.union(prev_positive_keywords)
            ),
            *positive_keypath,
        )

        prev_negative_keypath = (*prev_config, self.data.negative_key)
        negative_keypath = (*config, self.data.negative_key)
        transaction.kset(
            self.data.keywords,
            copy.deepcopy(
                self.negative_labelled_keywords.union(
                    transaction.rget(self.data.keywords, *prev_negative_keypath)
                )
            ),
            *negative_keypath,
        )

        labelled_keypath = (*config, "labelled")
        transaction.kset(
            self.labelled_keywords_results,
            copy.deepcopy(
                self.positive_labelled_keywords.union(prev_positive_keywords)
            ),
            *labelled_keypath,
        )

        candidates_keypath = (*config, "candidates")
        transaction.kset(
            self.candidate_results,
            copy.deepcopy(self.candidate_keywords),
            *candidates_keypath,
        )

        # keep track of how many tweets & user have been labelled
        tweet_record_keypath = (*config, "tweets")
        prev_tweet_record_keypath = (*prev_config, "tweets")
        iter_tweets = self.data.get_current_tweet_set(self.source_hashtag_dict).union(
            transaction.rget(self.labelled_keywords_results, *prev_tweet_record_keypath)
        )
        transaction.kset(
            self.labelled_keywords_results,
            iter_tweets,
            *tweet_record_keypath,
        )
        tweet_coverage_keypath = (*config, "tweet_coverage")
        tweet_coverage = self.data.get_oracle_tweet_coverage(
            set([self.data.tweet_idx.get(t) for t in iter_tweets]),
            config.topic,
        )
        transaction.kset(
            self.labelled_keywords_results,
            tweet_coverage,
            *tweet_coverage_keypath,
        )

        user_record_keypath = (*config, "users")
        prev_user_record_keypath = (*prev_config, "users")
        iter_users = self.data.get_current_user_set(self.source_hashtag_dict).union(
            transaction.rget(self.labelled_keywords_results, *prev_user_record_keypath)
        )
        transaction.kset(
            self.labelled_keywords_results,
            iter_users,
            *user_record_keypath,
        )
        user_coverage_keypath = (*config, "user_coverage")
        user_coverage = self.data.get_oracle_user_coverage(
            set([self.data.user_idx.get(u) for u in iter_users]), config.topic
        )
        transaction.kset(
            self.labelled_keywords_results,
            user_coverage,
            *user_coverage_keypath,
        )

    def run_text_baseline(self, th_keypath: transaction.TweetHashtagKeyPath) -> set:
        tweet_ids = self.data.get_current_tweet_set(
            self.source_hashtag_dict, day=th_keypath.day
        )
        # print(
        #    f"running text baseline using day {th_keypath.day} and {self.data.iter_days-1}"
        # )
        tweets = self.text_data.get_tweets(tweet_ids, day=th_keypath.day)

        if th_keypath.method == "tinybert":
            top_keywords_fn = tinybert.get_top_keywords
        elif th_keypath.method == "tfidf":
            top_keywords_fn = tfidf.get_top_keywords
        elif th_keypath.method == "word2vec":
            top_keywords_fn = word2vec.get_top_keywords
        else:
            raise ValueError("Unknown baseline")

        return top_keywords_fn(
            tweets,
            th_keypath.budget,
            self.positive_labelled_keywords,
            self.negative_labelled_keywords,
        )

    def get_random_hashtag_scores(self, positive_labels: set, negative_labels: set):
        return {
            k: random.random()
            for k in self.hashtag_source_dict
            if (k not in positive_labels and k not in negative_labels)
        }

    def get_hashtag_scores(self, positive_labels: set, negative_labels: set):
        all_to_labelled_scores = {}
        for hashtag in self.hashtag_source_dict:
            if hashtag in positive_labels or hashtag in negative_labels:
                continue

            pos, pos_coef = 0, 0
            for positive_keyword in positive_labels:
                pos_inter_tids = self.hashtag_source_dict[hashtag].intersection(
                    self.hashtag_source_dict[positive_keyword]
                )
                pos += len(pos_inter_tids)
                pos_coef += 1

            neg, neg_coef = 0, 0
            for negative_keyword in negative_labels:
                neg_inter_tids = self.hashtag_source_dict[hashtag].intersection(
                    self.hashtag_source_dict[negative_keyword]
                )
                neg += len(neg_inter_tids)
                neg_coef += 1

            all_to_labelled_scores[hashtag] = pos / max(1, pos_coef) - neg / max(
                1, neg_coef
            )

        return all_to_labelled_scores

    def run_text_experiment_iterations(
        self, top_keywords: list, config: transaction.Config, oracle_keywords: set
    ):
        self.candidate_keywords.update(set(top_keywords))
        assert len(top_keywords) <= config.budget, "Too many candidate keywords"

        for candidate_keyword in self.candidate_keywords:
            if candidate_keyword in oracle_keywords:
                self.positive_labelled_keywords.add(candidate_keyword)
            else:
                self.negative_labelled_keywords.add(candidate_keyword)

        return True

    def run_experiment_iteration(self, oracle_keywords: set):
        try:
            _, candidate_keyword = self.topic_queue.get_nowait()
        except queue.Empty:
            return False

        self.candidate_keywords.add(candidate_keyword)
        self.budget -= 1

        if candidate_keyword in oracle_keywords:
            self.positive_labelled_keywords.add(candidate_keyword)
            self.update_queue(candidate_keyword)
        else:
            self.negative_labelled_keywords.add(candidate_keyword)

        return True

    def init_experiment_sets(
        self, positive_available_hashtags: set, negative_available_hashtags: set
    ):
        self.candidate_keywords, self.queued_keywords = set(), set()
        self.positive_labelled_keywords = positive_available_hashtags
        self.negative_labelled_keywords = negative_available_hashtags

    def init_topic_queue(self) -> queue.PriorityQueue:
        self.topic_queue = queue.PriorityQueue()
        for keyword in self.positive_labelled_keywords:
            self.update_queue(keyword)

        return self.topic_queue

    def update_queue(self, keyword: str):
        neighbours = (
            self.get_hashtag_neighbours(
                keyword, self.hashtag_source_dict, self.source_hashtag_dict
            )
            - self.positive_labelled_keywords
            - self.negative_labelled_keywords
            - self.queued_keywords
        )
        self.load_queue(neighbours)

    def load_queue(self, neighbours: set, max_candidates: int = 30) -> set:
        sorted_neighbours_by_centralities = {
            k: v
            for k, v in sorted(
                self.centralities.items(), key=lambda item: item[1], reverse=True
            )
            if k in neighbours
        }
        for neighbour in list(sorted_neighbours_by_centralities):
            self.queued_keywords.add(neighbour)
            self.topic_queue.put((-self.centralities[neighbour], neighbour))

    def get_hashtag_neighbours(
        self, keyword: str, hashtag_source_dict: dict, source_hashtag_dict: dict
    ) -> set:
        neighbours = set()
        if keyword not in hashtag_source_dict:
            return neighbours

        for tid in hashtag_source_dict[keyword]:
            neighbours.update(source_hashtag_dict[tid])

        return neighbours

    def save_results(self):
        self.output = transaction.rinit(self.data.day_configs)
        self.get_precision()
        self.get_recall()
        self.save_coverage()
        self.save_volume()

        with open(self.data.results_file, "w") as f:
            json.dump(self.output, f)

    def save_volume(self):
        print("Saving volume...")
        volume_concerns = ["tweets", "users"]
        for config in tqdm(
            map(lambda x: transaction.Config(*x), self.data.day_configs)
        ):
            if not transaction.rget(self.labelled_keywords_results, *config):
                continue

            for volume_concern in volume_concerns:
                volume = len(
                    transaction.rget(
                        self.labelled_keywords_results, *(*config, volume_concern)
                    )
                )
                transaction.kset(self.output, volume, *(*config, volume_concern))

    def save_coverage(self):
        print("Saving user and tweet coverages...")
        coverage_concerns = ["user_coverage", "tweet_coverage"]
        for config in tqdm(
            map(lambda x: transaction.Config(*x), self.data.day_configs)
        ):
            if not transaction.rget(self.labelled_keywords_results, *config):
                continue

            for coverage_concern in coverage_concerns:
                coverage_val = transaction.rget(
                    self.labelled_keywords_results, *(*config, coverage_concern)
                )
                transaction.kset(
                    self.output, coverage_val, *(*config, coverage_concern)
                )

    def get_recall(self) -> dict:
        print("Saving recall...")
        for config in tqdm(
            map(lambda x: transaction.Config(*x), self.data.day_configs)
        ):
            if not transaction.rget(self.labelled_keywords_results, *config):
                continue

            oracle_keypath = (*config, self.data.oracle_key)
            oracle_keywords = transaction.rget(self.data.keywords, *oracle_keypath)
            seed_keypath = (*config, "seed")
            seed_keywords = transaction.rget(
                self.labelled_keywords_results, *seed_keypath
            )
            labelled_keypath = (*config, "labelled")
            result_keywords = transaction.rget(
                self.labelled_keywords_results, *labelled_keypath
            )

            recall = self.get_keyword_set_availability_share(
                oracle_keywords - seed_keywords, result_keywords - seed_keywords
            )
            transaction.kset(self.output, recall, *(*config, "recall"))

    def get_precision(self):
        print("Saving precision...")
        for config in tqdm(
            map(lambda x: transaction.Config(*x), self.data.day_configs)
        ):
            if not transaction.rget(self.labelled_keywords_results, *config):
                continue

            active_oracle_keypath = (*config, "active_oracles")
            active_oracle_keywords = transaction.rget(
                self.candidate_results, *active_oracle_keypath
            )

            candidate_keypath = (*config, "candidates")
            candidate_keywords = transaction.rget(
                self.candidate_results, *candidate_keypath
            )

            precision = self.get_keyword_set_availability_share(
                candidate_keywords, active_oracle_keywords
            )
            transaction.kset(self.output, precision, *(*config, "precision"))

    def get_keyword_set_availability_share(
        self, container: set, contained: set
    ) -> float:
        return (
            len(set(container).intersection(contained)) / len(container)
            if len(container) > 0
            else None
        )
