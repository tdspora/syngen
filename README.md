# Syngen

Syngen is an unsupervised tabular data generation tool. It is useful for generation of test data with a given table as a template. Most datatypes including floats, integers, datetime, text, categorical, binary are supported. The linked tables i.e., tables sharing a key can also be generated using the simple statistical approach.

The tool is based on the variational autoencoder model (VAE). The Bayesian Gaussian Mixture model is used to further detangle the latent space.

## Getting started

Use pip to install the library:

`pip install syngen`

The training and inference processes are separated with two cli entry points. The training one receives paths to the original table, metadata json file or table name and used hyperparameters. To start training with the sensible defaults run

`train --source PATH_TO_ORIGINAL_CSV --table_name TABLE_NAME`

This will train a model and save the model artifacts to disk.

To generate data simply call

`infer SIZE TABLE_NAME`

This will create a csv file with the synthetic table in ./model_artifacts/tmp_store/TABLE_NAME/merged_infer_TABLE_NAME.csv

Here is a quick example:

```
pip install syngen
train --source ./data/Churn_modelling.csv –-table_name Churn
infer 5000 Churn
```

## Features

### Training

You can add flexibility to the training and inference processes using additional hyperparameters. For training single table call:

`train --source PATH_TO_ORIGINAL_CSV --table_name TABLE_NAME --epochs INT --row_limit INT --drop_null BOOL`

- source – a path to the csv table that you want to use a reference
- table_name – an arbitrary string to name the directories 
- epochs – the number of training epochs. Since the early stopping mechanism is implemented the bigger is the better
- row_limit – the number of rows to train over. A number less then the original table length will randomly subset the specified rows number
- drop_null – whether to drop rows with at least one missing value

For training the multiple linked tables (see below) call:

`train --metadata_path PATH_TO_METADATA_YAML`

- metadata_path – a path to the json file containing the metadata for linked tables generation


### Inference (generation)

You can customize the inference processes by calling for one table:

`infer --size INT --table_name STR --run_parallel BOOL --batch_size INT --random_seed INT --print_report BOOL`
 
- size - the desired number of rows to generate
- table_name – the name of the table, same as in training
- run_parallel – whether to use multiprocessing (feasible for tables > 5000 rows)
- batch_size – if specified, the generation is split into batches. This can save the RAM
- random_seed – if specified, generates a reproducible result
- print_report – whether to generate plots of pairwise distributions, accuracy matrix and print the median accuracy
 
For linked tables you can simply call:

`infer --metadata_path PATH_TO_METADATA`
 
- metadata_path – a path to metadata yaml file to generate linked tables

The metadata can contain any of the arguments above for each table. If so, the duplicated arguments from the CLI 
will be ignored.



### Linked tables generation

To generate linked tables, you should provide metadata in yaml format. It is used to handle complex 
relations for any number of tables. You can also specify additional parameters needed for training and inference. In 
this case they will be ignored in the CLI call.

The yaml metadata file should match the following template:

    CUSTOMER:                                       # Table name
        source: "s3://syn-gen/files/customer.csv"   # Supported formats include cloud storage locations, local files
                 
        train_settings:                             # Settings for training process
            epochs: 10                              # Number of epochs if different from the default in the command line options
            drop_null: true                         # Drop rows with NULL values
                 
        infer_settings:                             # Settings for infer process
            size: 500                               # Size for generated data
            run_parallel: True                      # Turn on or turn off parallel training process
            print_report: True                      # Turn on or turn off generation of the report
        keys:
            PK_CUSTOMER_ID:                         # Name of a key. Only one PK per table.
                type: "PK"                          # The key type. Supported: PK - primary key, FK - foreign key, TKN - token key
                columns:                            # Array of column names
                    - customer_id
     
            UQ1:                                    # Name of a key
                type: "UQ"                          # One or many unique keys
                columns:
                    - e_mail
     
            FK1:                                    # One or many foreign keys
                type: "FK"
                columns:                            # Array of columns in the source table
                    - e_mail
                    - alias
                references:
                    table: "PROFILE"                # Name of the target table
                    columns:                        # Array of columns in the target table
                        - e_mail
                        - alias
       
            FK2:
                type: "FK"
                columns:
                    - address_id
                references:
                    table: "ADDRESS"
                    columns:
                        - address_id

     
    ORDER:
        source: "./files/order.csv"
     
        train_settings:
            epochs: 10                              # Number of epochs if different from the default in the command line options
            drop_null: true                         # Drop rows with NULL values
     
        infer_settings:                             # Settings for infer process
            size: 500                               # Size for generated data
            run_parallel: True                      # Turn on or turn off parallel training process
            print_report: True                      # Turn on or turn off generation of the report
        keys:
            pk_order_id:
                type: "PK"
                columns:
                    - order_id
     
            FK1:
                type: "FK"
                columns:
                    - customer_id
                references:
                    table: "CUSTOMER"
                    columns:
                        - customer_id
For related tables training you can use the commands:

```
train --metadata_path=PATH_TO_YAML_METADATA_FILE
infer --metadata_path=PATH_TO_YAML_METADATA_FILE
```

If `--metadata_path` is present and the metadata contains the necessary parameters, other CLI parameters will be ignored.

### Docker images using

The train and inference components of Syngen is available as public docker images:

<https://hub.docker.com/r/tdspora/syngen-train>

<https://hub.docker.com/r/tdspora/syngen-infer>

To run dockerized code (see parameters description in *Training* and *Inference* sections) for one table call:

```
docker pull tdspora/syngen-train:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train --table_name=TABLE_NAME --source=./model_artifacts/YOUR_CSV_FILE.csv

docker pull tdspora/syngen-infer:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer --size=NUMBER_OF_ROWS --table_name=TABLE_NAME
```

PATH_TO_LOCAL_FOLDER is an absolute path to the folder where your original csv is stored.

You can add any arguments listed in the corresponding sections for infer and training processes.

To run dockerized code for linked tables simply call:

```
docker pull tdspora/syngen-train:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker pull tdspora/syngen-infer:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

You can add any arguments listed in the corresponding sections for infer and training processes, however, they will be 
overwrited by corresponding arguments in the metadata file.