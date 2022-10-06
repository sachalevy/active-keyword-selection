# Baselines


## Recovering tweets from disk storage
We hydrated our tweets using our in-house data collection pipeline, `snpipeline`, which is directly stores tweets to a MongoDB server instance after parsing them from relations (retweet, mention, hashtag, quote, etc.).

<details>
    <summary>More context</summary>
    For the purpose of these experiments, we not only retrieve the `hashtag_relations` collection from our database, but also the tweet texts, stored in the `tweets` collection. Similarly to Bozarth et al. (see paper for citation), we only retrieve tweets with original text content from its users (we skip all retweets).

    In the `keyselect.baselines.utils.py` file is implemented a script to download in parallel partitions from this tweet database, parse their texts on the fly to conform to our analysis, and store them in lightweight no-sql databases (implemented using the pythonic `pickledb`). Since we're not sure about how thread-safe is `pickledb`, we dedicate databases per day and partition ids between the parallel processes. We then implement another script to merge the retrieved databases into single day-wise databases, and merge the `user_tweet_{i}.json` files into a single json file containing all the relations.
</details>

To retrieve the tweets, follow our hydration tutorial presented on the `snpipeline` package. Start by installing it:
```bash
pip install snpipeline
```

Next, run the util script to download the tweets from your database (make sure you're in the root directory of the `keyselect` project):
```bash
bash scripts/download_tweets_from_mongodb.sh
```
> This script downloads all tweets from the database and arranges them into individual `pickledb` files.

