# Active Keyword Selection for Tracking Evolving Topics on Twitter

[![arXiv](https://img.shields.io/badge/arXiv-2209.11135-b31b1b)](https://arxiv.org/abs/2209.11135)
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

Source code for our paper `Active Keyword Selection for Tracking Evolving Topics on Twitter`, including our method `KeySelect`.

## Quick Start

To get a quick demonstration of how to use `KeySelect` to find topic-relevant hashtags within a corpus of tweets, check out our [online demo](https://github.com/sachalevy/diskeyword).

Otherwise, first clone this repository before setting up your python environment:
```bash
python3 -m venv env/
pip install -r requirements.txt
pip install -e .
```

## Experiment Reproduction

### Dataset

You'll need to download the data prior to reproducing the experiments. Twitter only allows us to share the tweet ids (not their content), so you'll need to hydrate them yourselves. You may also reach out to us (sacha.levy@mail.mcgill.ca) to request the full data.

Use the following script to download and hydrate the tweets:
```bash
python3 src/utils/download.py
```
> This script first downloads the hashtag relations and tweet ids files from online storage, and then hydrates the tweets before storing them to a json file.


### Execution

Then, run the experiments with:
```bash
python3 src/main.py
```

Run the Jupyter Notebook `src/experiments/Figure.ipynb` to reproduce the figures presented in the paper.


## Maintainers

- Sacha Lévy (sacha.levy@mail.mcgill.ca)
