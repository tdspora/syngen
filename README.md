# Syngen

Syngen is an unsupervised tabular data generation tool. It is useful for generation of test data with a given table as a template. Most datatypes including floats, integers, datetime, text, categorical, binary are supported. The linked tables i.e., tables sharing a key can also be generated using the simple statistical approach. 
The source of data might be in CSV, Avro format and should be located locally.

The tool is based on the variational autoencoder model (VAE). The Bayesian Gaussian Mixture model is used to further detangle the latent space.

## Getting started

Use pip to install the library:

`pip install syngen`

The training and inference processes are separated with two cli entry points. The training one receives paths to the original table, metadata json file or table name and used hyperparameters.<br>

To start training with defaults parameters run:

`train --source PATH_TO_ORIGINAL_CSV --table_name TABLE_NAME`

This will train a model and save the model artifacts to disk.

To generate with defaults parameters data simply call:

`infer --table_name TABLE_NAME`

<i>Please notice that the name should match the one you used in the training process.</i><br>
This will create a csv file with the synthetic table in <i>./model_artifacts/tmp_store/TABLE_NAME/merged_infer_TABLE_NAME.csv</i>.<br>

Here is a quick example:

```
pip install syngen
train --source ./example-data/housing.csv –-table_name Housing
infer --table_name Housing
```
As the example you can use the dataset <i>"Housing"</i> in [example-data/housing.csv](example-data/housing.csv).
In this example, our real-world data is <a href="https://www.kaggle.com/datasets/camnugent/california-housing-prices" target="_blank">"Housing"</a> from Kaggle.

## Features

### Training

You can add flexibility to the training and inference processes using additional hyperparameters.<br>
For training of single table call:

`train --source PATH_TO_ORIGINAL_CSV --table_name TABLE_NAME --epochs INT --row_limit INT --drop_null BOOL`

For training of the multiple linked tables call:

`train --metadata_path PATH_TO_METADATA_YAML`

The parameters which you can set up for training process:

- <i>source</i> – required parameter for training of single table, a path to the file that you want to use as a reference
- <i>table_name</i> – required parameter for training of single table, an arbitrary string to name the directories 
- <i>epochs</i> – a number of training epochs. Since the early stopping mechanism is implemented the bigger value of epochs is the better
- <i>row_limit</i> – a number of rows to train over. A number less than the original table length will randomly subset the specified number of rows
- <i>drop_null</i> – whether to drop rows with at least one missing value
- <i>metadata_path</i> – a path to the json file containing the metadata for linked tables generation

Requirements for parameters of training process:
* <i>source</i> - data type - string
* <i>table_name</i> - data type - string
* <i>epochs</i> - data type - integer, must be equal to or more than 1, default value is 10
* <i>drop_null</i> - data type - boolean, default value - False
* <i>row_limit</i> - data type - integer
* <i>metadata_path</i> - data type - string


### Inference (generation)

You can customize the inference processes by calling for one table:

`infer --size INT --table_name STR --run_parallel BOOL --batch_size INT --random_seed INT --print_report BOOL`
 
For linked tables you can simply call:

`infer --metadata_path PATH_TO_METADATA`

The parameters which you can set up for generation process:

- <i>size</i> - the desired number of rows to generate
- <i>table_name</i> – required parameter for inference of single table, the name of the table, same as in training
- <i>run_parallel</i> – whether to use multiprocessing (feasible for tables > 5000 rows)
- <i>batch_size</i> – if specified, the generation is split into batches. This can save the RAM
- <i>random_seed</i> – if specified, generates a reproducible result
- <i>print_report</i> – whether to generate plots of pairwise distributions, accuracy matrix and print the median accuracy
- <i>metadata_path</i> – a path to metadata yaml file to generate linked tables

Requirements for parameters of generation process:
* <i>size</i> - data type - integer, must be equal to or more than 1, default value is 100
* <i>table_name</i> - data type - string
* <i>run_parallel</i> - data type - boolean, default value is False
* <i>batch_size</i> - data type - integer, must be equal to or more than 1
* <i>random_seed</i> - data type - integer
* <i>print_report</i> - data type - boolean, default value is False
* <i>metadata_path</i> - data type - string

The metadata can contain any of the arguments above for each table. If so, the duplicated arguments from the CLI 
will be ignored.


### Linked tables generation

To generate linked tables, you should provide metadata in yaml format. It is used to handle complex 
relations for any number of tables. You can also specify additional parameters needed for training and inference in the metadata file 
and in this case, they will be ignored in the CLI call.

The yaml metadata file should match the following template:

    CUSTOMER:                                       # Table name
        source: "./files/customer.csv"              # Supported formats include local files in CSV, Avro formats
                 
        train_settings:                             # Settings for training process
            epochs: 10                              # Number of epochs if different from the default in the command line options
            drop_null: true                         # Drop rows with NULL values
            row_limit: 1000                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number
                 
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
                columns:                            # Array of columns in the current table
                    - e_mail
                    - alias
                references:
                    table: "PROFILE"                # Name of the parent table
                    columns:                        # Array of columns in the parent table
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
            row_limit: 1000                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number
     
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

<i>You can find the example of metadata file in [example-metadata/housing_metadata.yaml](example-metadata/housing_metadata.yaml)</i><br>

For related tables training you can use the commands:

```
train --metadata_path=PATH_TO_YAML_METADATA_FILE
infer --metadata_path=PATH_TO_YAML_METADATA_FILE
```
Here is a quick example:

```
train --metadata_path="./example-metadata/housing_metadata.yaml"
infer --metadata_path="./example-metadata/housing_metadata.yaml"
```

If `--metadata_path` is present and the metadata contains the necessary parameters, other CLI parameters will be ignored.<br>

### Docker images

The train and inference components of <i>syngen</i> is available as public docker images:

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

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call.

To run dockerized code for linked tables simply call:

```
docker pull tdspora/syngen-train:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker pull tdspora/syngen-infer:latest
docker run --rm -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call, however, they will be 
overwritten by corresponding arguments in the metadata file.
