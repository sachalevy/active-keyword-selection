import json

from src import loader, runner
from src.baselines.utils import FastTweetLoader


def main():
    use_users = True  # toggle this parameter to use the tweet/user-hashtag graph
    dump_filepath = (
        loader.DEFAULT_DUMP_FILEPATH
        if not use_users
        else loader.DEFAULT_USERS_DUMP_FILEPATH
    )
    print(f"Dumping experiment to {dump_filepath}")

    if dump_filepath.is_file():
        data = loader.TemporalExperimentDataLoader.load(use_users)
        print("Loaded data from dump file, summary:")
        print(
            json.dumps(
                {
                    "user count": data.user_count,
                    "tweet count": data.tweet_count,
                    "hashtag count": data.hashtag_count,
                    "configurations": len(data.day_configs),
                    "oracle tweet counts": {
                        topic: len(data.oracle_tweets.get(topic))
                        for topic in data.experiment_topics
                    },
                    "oracle user counts": {
                        topic: len(data.oracle_users.get(topic))
                        for topic in data.experiment_topics
                    },
                },
                indent=4,
            )
        )
    else:
        # download file from google drive if not there
        if not loader.LOCAL_ECHEN_FILEPATH.is_file():
            assert (
                loader.LOCAL_ECHEN_FILEPATH.parent.is_dir()
            ), "Please make sure the data directory exists"

        data = loader.TemporalExperimentDataLoader(
            data_filepath=loader.LOCAL_ECHEN_FILEPATH,
            experiment_topics=["MASK", "VACCINE", "LOCKDOWN"],
            use_users=use_users,
            keyword_sampling_methods=[
                "ours",
                "random",
                "degree",
                "tfidf",
                "word2vec",
            ],
            keyword_sampling_counts=[10],
            evaluation_budgets=[3, 10, 30],
        )
        data.dump()

    text_data = FastTweetLoader()
    exp = runner.TemporalExperimentRunner(data, text_data)
    exp.run_experiments()
    exp.save_results()


if __name__ == "__main__":
    main()
