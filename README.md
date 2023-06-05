# How Fragile is Relation Extraction under Entity Replacements?
[![arXiv](https://img.shields.io/badge/arXiv-2305.13551-b31b1b.svg)](https://arxiv.org/abs/2305.13551)

## Overview

This repo includes the open-sourced code and data for our work *How Fragile is Relation Extraction under Entity Replacements?*. In this work, we operate the random and type-constrained entity replacements over the RE instances in TACRED and evaluate the state-of-the-art RE models under the entity replacements. We observe the 30\% - 50\% F1 score drops on the state-of-the-art RE models under entity replacements.

Relation extraction models are fragile to our entity replacements (ENTRE)|  
:-------------------------:|
![](Fig/fig_5.svg)|

## A New Relation Extraction Dataset: ENTRED

[2023/05/29] ENTRED is a challenging relation extraction dataset that we build by applying the type-constrained entity replacements on TACRED. You need not run the code from scratch to build the dataset ENTRED from beginning. We have provided the built ENTRED at `new_test.json`. We introduce all the .json files as following.

1. `test_entred.json`: The proposed benchmark: ENTRED.
2. `test.json`: The test set of TACRED.
3. `re_test.json`: The test set of Re-TACRED.
4. `rev_test.json`: The test set of TACREV.

## Run

Evaluation of relation extraction with entity replacements.

### Collecting Person and Organization Names from Wikipedia (Optional)
>get_wiki.ipynb

This step can be skipped because we have stored the outputs to `wiki_organization.output` and `wiki_person.output`.

### Data Analysis of Existing Relation Extraction Datasets (Optional)
>entre.ipynb

This step can be skipped because we have stored the outputs to `final_id_resample_ls.output`.

### Evaluation of LUKE under entity replacements
>python entre.py

## Evaluate LUKE on TACRED and ENTRED 

### Evaluate LUKE on TACRED:
>python luke.py --input_file test.json --output_file luke_pred.output

### Evaluate LUKE on ENTRED:
>python luke.py --input_file test_entred.json --output_file new_luke_pred.output

## License

This project is licensed under the Apache-2.0 License.