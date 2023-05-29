
# How Fragile is Relation Extraction under Entity Replacements?

## Overview

This repo includes the open-sourced code and data for our work *How Fragile is Relation Extraction under Entity Replacements?*.

## Dataset

[2023/05/29] We provide the test set of TACRED, TACREV, and Re-TACRED for the ease of evaluation.

2. `test.json`: The test set of TACRED.
3. `re_test.json`: The test set of Re-TACRED.
4. `rev_test.json`: The test set of TACREV.

## Run

The running of the evaluation of TACRED with entity replacements.

### Collecting Person and Organization Names from Wikipedia (Optional)
>get_wiki.ipynb

This step can be skipped because we have stored the outputs to `wiki_organization.output` and `wiki_person.output`.

### Data Analysis (Optional)
>eric.ipynb

This step can be skipped because we have stored the outputs to `final_id_resample_ls.output`.

### Evaluation of LUKE under entity replacements
>python entre.py

## License

This project is licensed under the Apache-2.0 License.