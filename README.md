[![CI/CD](https://github.com/tdspora/syngen/actions/workflows/action-build-deploy.yml/badge.svg?branch=main)](https://github.com/tdspora/syngen/actions/workflows/action-build-deploy.yml)

# EPAM Syngen

EPAM Syngen is an unsupervised tabular data generation tool. It is useful for generation of test data with a given table as a template. Most datatypes including floats, integers, datetime, text, categorical, binary are supported. The linked tables i.e., tables sharing a key can also be generated using the simple statistical approach. 
The source of data might be in CSV, Avro and Excel format and should be located locally and be in UTF-8 encoding.

The tool is based on the variational autoencoder model (VAE). The Bayesian Gaussian Mixture model is used to further detangle the latent space.

## Getting started

Use pip to install the library:

`pip install syngen`

The training and inference processes are separated with two cli entry points. The training one receives paths to the original table, metadata json file or table name and used hyperparameters.<br>

To start training with defaults parameters run:

```bash
train --source PATH_TO_ORIGINAL_CSV \
    --table_name TABLE_NAME
```

This will train a model and save the model artifacts to disk.

To generate with defaults parameters data simply call:

```bash
infer --table_name TABLE_NAME
```

<i>Please notice that the name should match the one you used in the training process.</i><br>
This will create a csv file with the synthetic table in <i>./model_artifacts/tmp_store/TABLE_NAME/merged_infer_TABLE_NAME.csv</i>.<br>

Here is a quick example:

```bash
pip install syngen
train --source ./examples/example-data/housing.csv –-table_name Housing
infer --table_name Housing
```
As the example you can use the dataset <i>"Housing"</i> in [examples/example-data/housing.csv](examples/example-data/housing.csv).
In this example, our real-world data is <a href="https://www.kaggle.com/datasets/camnugent/california-housing-prices" target="_blank">"Housing"</a> from Kaggle.

## Features

### Training

You can add flexibility to the training and inference processes using additional hyperparameters.<br>
For training of single table call:

```bash
train --source PATH_TO_ORIGINAL_CSV \
    --table_name TABLE_NAME \
    --epochs INT \
    --row_limit INT \
    --drop_null BOOL \
    --print_report BOOL \
    --batch_size INT
```

To train one or more tables using a metadata file, you can use the following command:

```bash
train --metadata_path PATH_TO_METADATA_YAML
```

The parameters which you can set up for training process:

- <i>source</i> – required parameter for training of single table, a path to the file that you want to use as a reference
- <i>table_name</i> – required parameter for training of single table, an arbitrary string to name the directories 
- <i>epochs</i> – a number of training epochs. Since the early stopping mechanism is implemented the bigger value of epochs is the better
- <i>row_limit</i> – a number of rows to train over. A number less than the original table length will randomly subset the specified number of rows
- <i>drop_null</i> – whether to drop rows with at least one missing value
- <i>batch_size</i> – if specified, the training is split into batches. This can save the RAM
- <i>print_report</i> - whether to generate accuracy and sampling reports. Please note that the sampling report is generated only if the `row_limit` parameter is set.
- <i>metadata_path</i> – a path to the metadata file containing the metadata
- <i>column_types</i> - might include the section <i>categorical</i> which contains the listed columns defined as categorical by a user

Requirements for parameters of training process:
* <i>source</i> - data type - string
* <i>table_name</i> - data type - string
* <i>epochs</i> - data type - integer, must be equal to or more than 1, default value is 10
* <i>row_limit</i> - data type - integer
* <i>drop_null</i> - data type - boolean, default value - False
* <i>batch_size</i> - data type - integer, must be equal to or more than 1, default value - 32
* <i>print_report</i> - data type - boolean, default value is False
* <i>metadata_path</i> - data type - string
* <i>column_types</i> - data type - dictionary with the key <i>categorical</i> - the list of columns (data type - string)


### Inference (generation)

You can customize the inference processes by calling for one table:

```bash
infer --size INT \
    --table_name STR \
    --run_parallel BOOL \
    --batch_size INT \
    --random_seed INT \
    --print_report BOOL
```
 
To generate one or more tables using a metadata file, you can use the following command:

```bash
infer --metadata_path PATH_TO_METADATA
```

The parameters which you can set up for generation process:

- <i>size</i> - the desired number of rows to generate
- <i>table_name</i> – required parameter for inference of single table, the name of the table, same as in training
- <i>run_parallel</i> – whether to use multiprocessing (feasible for tables > 5000 rows)
- <i>batch_size</i> – if specified, the generation is split into batches. This can save the RAM
- <i>random_seed</i> – if specified, generates a reproducible result
- <i>print_report</i> – whether to generate accuracy and sampling reports. Please note that the sampling report is generated only if the row_limit parameter is set.
- <i>metadata_path</i> – a path to metadata file

Requirements for parameters of generation process:
* <i>size</i> - data type - integer, must be equal to or more than 1, default value is 100
* <i>table_name</i> - data type - string
* <i>run_parallel</i> - data type - boolean, default value is False
* <i>batch_size</i> - data type - integer, must be equal to or more than 1
* <i>random_seed</i> - data type - integer, must be equal to or more than 0
* <i>print_report</i> - data type - boolean, default value is False
* <i>metadata_path</i> - data type - string

The metadata can contain any of the arguments above for each table. If so, the duplicated arguments from the CLI 
will be ignored.

<i>Note:</i> If you want to set the logging level, you can use the parameter <i>log_level</i> in the CLI call:

```bash
train --source STR --table_name STR --log_level STR
train --metadata_path STR --log_level STR
infer --size INT --table_name STR --log_level STR
infer --metadata_path STR --log_level STR
```

where <i>log_level</i> might be one of the following values: <i>DEBUG, INFO, WARNING, ERROR, CRITICAL</i>.


### Linked tables generation

To generate one or more tables, you might provide metadata in yaml format. By providing information about the relationships 
between tables via metadata, it becomes possible to manage complex relationships across any number of tables. 
You can also specify additional parameters needed for training and inference in the metadata file and in this case, 
they will be ignored in the CLI call.

<i>Note:</i> By using metadata file, you can also generate tables with absent relationships. 
In this case, the tables will be generated independently.

The yaml metadata file should match the following template:
```yaml
global:                                     # Global settings. Optional parameter. In this section you can specify training and inference settings which will be set for all tables
  train_settings:                           # Settings for training process. Optional parameter
    epochs: 10                              # Number of epochs if different from the default in the command line options. Optional parameter
    drop_null: False                        # Drop rows with NULL values. Optional parameter
    row_limit: null                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number. Optional parameter
    batch_size: 32                          # If specified, the training is split into batches. This can save the RAM. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    
  infer_settings:                           # Settings for infer process. Optional parameter
    size: 100                               # Size for generated data. Optional parameter
    run_parallel: False                     # Turn on or turn off parallel training process. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    batch_size: null                        # If specified, the generation is split into batches. This can save the RAM. Optional parameter
    random_seed: null                       # If specified, generates a reproducible result. Optional parameter

CUSTOMER:                                   # Table name. Required parameter               
  train_settings:                           # Settings for training process. Required parameter
    source: "./files/customer.csv"          # The path to the original data. Supported formats include local files in CSV, Avro formats. Required parameter
    epochs: 10                              # Number of epochs if different from the default in the command line options. Optional parameter
    drop_null: False                        # Drop rows with NULL values. Optional parameter
    row_limit: null                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number. Optional parameter
    batch_size: 32                          # If specified, the training is split into batches. This can save the RAM. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    column_types:
      categorical:                          # Force listed columns to have categorical type (use dictionary of values). Optional parameter
        - gender
        - marital_status
        
  format:                                   # Settings for reading and writing data in 'csv', 'psv', 'tsv', 'txt', 'xls', 'xlsx' format. Optional parameter
    sep: ','                                # Delimiter to use. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    quotechar: '"'                          # The character used to denote the start and end of a quoted item. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    quoting: minimal                        # Control field quoting behavior per constants - ["all", "minimal", "non-numeric", "none"]. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    escapechar: '"'                         # One-character string used to escape other characters. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    encoding: null                          # A string representing the encoding to use in the output file. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    header: infer                           # Row number(s) to use as the column names, and the start of the data. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats 
    skiprows: null                          # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    on_bad_lines: error                     # Specifies what to do upon encountering a bad line (a line with too many fields) - ["error", "warn", "skip"]. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    engine: null                            # Parser engine to use - ["c", "python"]. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    na_values: null                         # Additional strings to recognize as NA/NaN. The first value of the array will be used to replace NA/NaN values. Optional parameter. Applicable for 'csv', 'psv', 'tsv', 'txt' formats
    sheet_name: 0                           # Name of the sheet in the Excel file. Optional parameter. Applicable for 'xls', 'xlsx' formats
  infer_settings:                           # Settings for infer process. Optional parameter
    destination: "./files/generated_data_customer.csv" # The path where the generated data will be stored. Supported formats include local files in CSV, Avro formats. Required parameter
    size: 100                               # Size for generated data. Optional parameter
    run_parallel: False                     # Turn on or turn off parallel training process. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    batch_size: null                        # If specified, the generation is split into batches. This can save the RAM. Optional parameter
    random_seed: null                       # If specified, generates a reproducible result. Optional parameter
  keys:                                     # Keys of the table. Optional parameter
    PK_CUSTOMER_ID:                         # Name of a key. Only one PK per table.
      type: "PK"                            # The key type. Supported: PK - primary key, FK - foreign key, TKN - token key
      columns:                              # Array of column names
        - customer_id
     
    UQ1:                                    # Name of a key
      type: "UQ"                            # One or many unique keys
      columns:
        - e_mail
     
    FK1:                                    # One or many foreign keys
      type: "FK"
      columns:                              # Array of columns in the current table
        - e_mail
        - alias
      references:
        table: "PROFILE"                    # Name of the parent table
        columns:                            # Array of columns in the parent table
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

     
ORDER:                                      # Table name. Required parameter    
  train_settings:                           # Settings for training process. Required parameter
    source: "./files/order.csv"             # The path to the original data. Supported formats include local files in CSV, Avro formats. Required parameter
    epochs: 10                              # Number of epochs if different from the default in the command line options. Optional parameter
    drop_null: False                        # Drop rows with NULL values. Optional parameter
    row_limit: null                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number. Optional parameter
    batch_size: 32                          # If specified, the training is split into batches. This can save the RAM. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    column_types:
    categorical:                            # Force listed columns to have categorical type (use dictionary of values). Optional parameter
      - gender
      - marital_status
     
  infer_settings:                           # Settings for infer process. Optional parameter
    destination: "./files/generated_data_order.csv" # The path where the generated data will be stored. Supported formats include local files in CSV, Avro formats. Required parameter
    size: 100                               # Size for generated data. Optional parameter
    run_parallel: False                     # Turn on or turn off parallel training process. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    batch_size: null                        # If specified, the generation is split into batches. This can save the RAM. Optional parameter
    random_seed: null                       # If specified, generates a reproducible result. Optional parameter
  format:                                   # Settings for reading and writing data in 'csv' format. Optional parameter
    sep: ','                                # Delimiter to use. Optional parameter
    quotechar: '"'                          # The character used to denote the start and end of a quoted item. Optional parameter
    quoting: minimal                        # Control field quoting behavior per constants - ["all", "minimal", "non-numeric", "none"]. Optional parameter
    escapechar: '"'                         # One-character string used to escape other characters. Optional parameter
    encoding: null                          # A string representing the encoding to use in the output file. Optional parameter
    header: infer                           # Row number(s) to use as the column names, and the start of the data. Optional parameter  
    skiprows: null                          # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. Optional parameter
    on_bad_lines: error                     # Specifies what to do upon encountering a bad line (a line with too many fields) - ["error", "warn", "skip"]. Optional parameter
    engine: null                            # Parser engine to use - ["c", "python"]. Optional parameter
    sheet_name: 0                           # Name of the sheet in the Excel file. Optional parameter
  keys:                                     # Keys of the table. Optional parameter
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
```
<i>Note:</i>In the section <i>"global"</i> you can specify training and inference settings for all tables. If the same settings are specified for a specific table, they will override the global settings.<br>

<i>You can find the example of metadata file in [examples/example-metadata/housing_metadata.yaml](examples/example-metadata/housing_metadata.yaml)</i><br>

By providing the necessary information through a metadata file, you can initiate training and inference processes using the following commands:

```bash
train --metadata_path=PATH_TO_YAML_METADATA_FILE
infer --metadata_path=PATH_TO_YAML_METADATA_FILE
```
Here is a quick example:

```bash
train --metadata_path="./examples/example-metadata/housing_metadata.yaml"
infer --metadata_path="./examples/example-metadata/housing_metadata.yaml"
```

If `--metadata_path` is present and the metadata contains the necessary parameters, other CLI parameters will be ignored.<br>

### Docker images

The train and inference components of <i>syngen</i> is available as public docker images:

<https://hub.docker.com/r/tdspora/syngen-train>

<https://hub.docker.com/r/tdspora/syngen-infer>

To run dockerized code (see parameters description in *Training* and *Inference* sections) for one table call:

```bash
docker pull tdspora/syngen-train:latest
docker run --rm \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train \
  --table_name=TABLE_NAME \
  --source=./model_artifacts/YOUR_CSV_FILE.csv

docker pull tdspora/syngen-infer:latest
docker run --rm \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer \
  --table_name=TABLE_NAME
```

PATH_TO_LOCAL_FOLDER is an absolute path to the folder where your original csv is stored.

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call.

To run dockerized code by providing the metadata file simply call:

```bash
docker pull tdspora/syngen-train:latest
docker run --rm \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker pull tdspora/syngen-infer:latest
docker run --rm \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call, however, they will be 
overwritten by corresponding arguments in the metadata file.

#### Logging level

Set the `LOGURU_LEVEL` environment variable to desired level of logging.
For example, to suppress the debug messages, add `-e LOGURU_LEVEL=INFO` to the `docker run` command:
```bash
docker pull tdspora/syngen-train:latest
docker run --rm -e LOGURU_LEVEL=INFO \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-train \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker pull tdspora/syngen-infer:latest
docker run --rm -e LOGURU_LEVEL=INFO \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen-infer \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

#### MLflow monitoring
Set the `MLFLOW_TRACKING_URI` environment variable to the desired MLflow tracking server, for instance:
http://localhost:5000/. You can also set `MLFLOW_ARTIFACTS_DESTINATION` environment variable to the preferred path 
(including cloud path), where the artifacts should be stored. When using Docker, ensure the environmental variables 
are set before running the container.

The provided environmental variables allow to track the training process, and the inference process, and store 
the artifacts in the desired location.
You can access the MLflow UI by navigating to the provided URL in your browser. If you store artifacts in remote storage,
ensure that all necessary credentials are provided before using Mlflow.

## Contribution

We welcome contributions from the community to help us improve and maintain our public GitHub repository. We appreciate any feedback, bug reports, or feature requests, and we encourage developers to submit fixes or new features using issues.

If you have found a bug or have a feature request, please submit an issue to our GitHub repository. Please provide as much detail as possible, including steps to reproduce the issue or a clear description of the feature request. Our team will review the issue and work with you to address any problems or discuss any potential new features.

If you would like to contribute a fix or a new feature, please submit a pull request to our GitHub repository. Please make sure your code follows our coding standards and best practices. Our team will review your pull request and work with you to ensure that it meets our standards and is ready for inclusion in our codebase.

We appreciate your contributions, and thank you for your interest in helping us maintain and improve our public GitHub repository.
