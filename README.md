# Multi-Domain Sentiment Analysis with BERT

This repository contains a script for downloading, processing, and preparing the Multi-Domain Sentiment Dataset for BERT-based sentiment analysis. The dataset includes reviews across multiple domains, enabling fine-grained analysis for various domains.

## Dataset Information

The Multi-Domain Sentiment Dataset contains reviews from several domains such as **books**, **electronics**, and **kitchen** items. Each review is labeled as either **positive** or **negative**, allowing for sentiment classification tasks.

**Source:** [Multi-Domain Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

## Requirements

- Python 3.6+
- [Transformers](https://huggingface.co/transformers/) library by Hugging Face
- Pandas
- Requests

Install all dependencies with:

```bash
pip install -r requirements.txt
```
## Running the Script

python preprocess_multi_domain.py

## Script Overview
. Download & Extract: Retrieves and unzips the dataset.
. Preprocess & Tokenize: Reads, labels, tokenizes, and splits data for BERT training.
. Save Preprocessed Data: Outputs train_multi_domain.csv and test_multi_domain.csv.
