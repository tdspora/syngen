![Build Status](https://github.com/tdspora/syngen/workflows/TESTING/badge.svg)
# Syngen

Syngen is an unsupervised tabular data generation tool. It is useful for generation of test data with a given table as a template. Most datatypes including floats, integers, datetime, text, categorical, binary are supported. The linked tables i.e., tables sharing a key can also be generated using the simple statistical approach.

The tool is based on the variational autoencoder model (VAE). The Bayesian Gaussian Mixture model is used to further detangle the latent space.

## Getting started

Use pip to install the library:

`pip install syngen`

The training and inference processes are separated with two cli entry points. The training one receives paths to the original table, metadata json file or table name and used hyperparameters. To start training with the sensible defaults run

`train PATH_TO_ORIGINAL_CSV –table_name TABLE_NAME`

This will train a model and save the model artifacts to disk.

To generate data simply call

`infer SIZE TABLE_NAME`

This will create a csv file with the synthetic table in ./model_artifacts/tmp_store/TABLE_NAME/merged_infer.csv

Here is a quick example:

```
pip install syngen
train ./data/Churn_modelling.csv –table_name Churn
infer 5000 Churn
```

## Features

### Training

You can add flexibility to the training and inference processes using additional hyperparameters.

`train PATH_TO_ORIGINAL_CSV –metadata_path PATH_TO_METADATA_JSON –table_name TABLE_NAME –epochs INT –row_limit INT –dropna BOOL –keys_mode BOOL`

- PATH_TO_ORIGINAL_CSV – a path to the csv table that you want to use a reference
- metadata_path – a path to the json file containing the metadata (see below)
- table_name – an arbitrary string to name the directories. If table name is provided and `–keys_mode` is False the `–metadata_path` argument is optional
- epochs – the number of training epochs. Since the early stopping mechanism is implemented the bigger is the better
- row_limit – the number of rows to train over. A number less then the original table length will randomly subset the specified rows number
- dropna – whether to drop rows with at least one missing value
- keys_mode – whether to train linked tables (see below)


### Inference

You can customize the inference processes by calling

`infer SIZE TABLE_NAME –run_parallel BOOL –batch_size INT –keys_mode BOOL –metadata_path PATH_TO_METADATA –random_seed INT- --print_report BOOL`

- SIZE - the desired number of rows to generate
- TABLE_NAME – the name of the table, same as in training
- run_parallel – whether to use multiprocessing (feasible for tables > 5000 rows)
- batch_size – if specified, the generation is split into batches. This can save the RAM
- keys_mode – whether to generate linked tables (see below)
- metadata_path – a path to metadata json file. If `--keys mode` is set to False the argument is optional
- random_seed – if specified, generates a reproducible result
- print_report – whether to generate plots of pairwise distributions, accuracy matrix and print the median accuracy


### Linked tables generation

To generate linked tables, you need to train tables in the special order:

A table with the Primary key (training) -> a table with the Primary key (inference) -> a table with the foreign key (training) -> a table with the foreign key (inference)

You have to set `--keys_mode` to True in every step and provide the metadata for the Foreign key table training and inference as a json file with the following structure:

`{"table_name": "NAME_OF_FK_TABLE", "fk": {"NAME_OF_FK_COLUMN": {"pk_table": "NAME_OF_PK_TABLE", "pk_column": "NAME_OF_PK_COLUMN (in PK table)"}}}`
