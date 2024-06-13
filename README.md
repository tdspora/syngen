[![CI/CD](https://github.com/tdspora/syngen/actions/workflows/action-build-deploy.yml/badge.svg?branch=main)](https://github.com/tdspora/syngen/actions/workflows/action-build-deploy.yml)

# EPAM Syngen

EPAM Syngen is an unsupervised tabular data generation tool. It is useful for generation of test data with a given table as a template. Most datatypes including floats, integers, datetime, text, categorical, binary are supported. The linked tables i.e., tables sharing a key can also be generated using the simple statistical approach.
The source of data might be in CSV, Avro and Excel format and should be located locally and be in UTF-8 encoding.

The tool is based on the variational autoencoder model (VAE). The Bayesian Gaussian Mixture model is used to further detangle the latent space.

## Prerequisites

Python 3.9 or Python 3.10 is required to run the library. The library is tested on Linux and Windows operating systems.
You can download Python from [the official website](https://www.python.org/downloads/) and install manually, or you can install Python [from your terminal](https://docs.python-guide.org/starting/installation/). After the installation of Python, please, check whether [pip is installed](https://pip.pypa.io/en/stable/getting-started/).

## Getting started

Before the installation of the library, you have to [set up the virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

You can install the library with CLI only:

```bash
pip install syngen
```

Otherwise, if you want to install the UI version with streamlit, run:

```bash
pip install syngen[ui]
```

*Note*: see details of the UI usage in the [corresponding section](#ui-web-interface)


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

where <i>log_level</i> might be one of the following values: <i>TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL</i>.


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
    get_infer_metrics: False                # Whether to fetch metrics for the inference process. If the parameter 'print_report' is set to True, the 'get_infer_metrics' parameter will be ignored and metrics will be fetched anyway. Optional parameter

CUSTOMER:                                   # Table name. Required parameter
  train_settings:                           # Settings for training process. Required parameter
    source: "./files/customer.csv"          # The path to the original data. Supported formats include local files in '.csv', '.avro' formats. Required parameter
    epochs: 10                              # Number of epochs if different from the default in the command line options. Optional parameter
    drop_null: False                        # Drop rows with NULL values. Optional parameter
    row_limit: null                         # Number of rows to train over. A number less than the original table length will randomly subset the specified rows number. Optional parameter
    batch_size: 32                          # If specified, the training is split into batches. This can save the RAM. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    column_types:
      categorical:                          # Force listed columns to have categorical type (use dictionary of values). Optional parameter
        - gender
        - marital_status

  format:                                   # Settings for reading and writing data in '.csv', '.psv', '.tsv', '.txt', '.xls', '.xlsx' format. Optional parameter
    sep: ','                                # Delimiter to use. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    quotechar: '"'                          # The character used to denote the start and end of a quoted item. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    quoting: minimal                        # Control field quoting behavior per constants - ["all", "minimal", "non-numeric", "none"]. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    escapechar: '"'                         # One-character string used to escape other characters. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    encoding: null                          # A string representing the encoding to use in the output file. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    header: infer                           # Row number(s) to use as the column names, and the start of the data. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    skiprows: null                          # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    on_bad_lines: error                     # Specifies what to do upon encountering a bad line (a line with too many fields) - ["error", "warn", "skip"]. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    engine: null                            # Parser engine to use - ["c", "python"]. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    na_values: null                         # Additional strings to recognize as NA/NaN. The first value of the array will be used to replace NA/NaN values. Optional parameter. Applicable for '.csv', '.psv', '.tsv', '.txt' formats
    sheet_name: 0                           # Name of the sheet in the Excel file. Optional parameter. Applicable for '.xls', '.xlsx' formats
  infer_settings:                           # Settings for infer process. Optional parameter
    destination: "./files/generated_data_customer.csv" # The path where the generated data will be stored. If the information about 'destination' isn't specified, by default the synthetic data will be stored locally in '.csv'. Supported formats include local files in '.csv', '.avro' formats. Optional parameter
    size: 100                               # Size for generated data. Optional parameter
    run_parallel: False                     # Turn on or turn off parallel training process. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    batch_size: null                        # If specified, the generation is split into batches. This can save the RAM. Optional parameter
    random_seed: null                       # If specified, generates a reproducible result. Optional parameter
    get_infer_metrics: False                # Whether to fetch metrics for the inference process. If the parameter 'print_report' is set to True, the 'get_infer_metrics' parameter will be ignored and metrics will be fetched anyway. Optional parameter
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
    source: "./files/order.csv"             # The path to the original data. Supported formats include local files in 'csv', '.avro' formats. Required parameter
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
    destination: "./files/generated_data_order.csv" # The path where the generated data will be stored. If the information about 'destination' isn't specified, by default the synthetic data will be stored locally in '.csv'. Supported formats include local files in 'csv', '.avro' formats. Required parameter
    size: 100                               # Size for generated data. Optional parameter
    run_parallel: False                     # Turn on or turn off parallel training process. Optional parameter
    print_report: False                     # Turn on or turn off generation of the report. Optional parameter
    batch_size: null                        # If specified, the generation is split into batches. This can save the RAM. Optional parameter
    random_seed: null                       # If specified, generates a reproducible result. Optional parameter
    get_infer_metrics: False                # Whether to fetch metrics for the inference process. If the parameter 'print_report' is set to True, the 'get_infer_metrics' parameter will be ignored and metrics will be fetched anyway. Optional parameter
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

<i>Note</i>:
<ul>
<li>In the section <i>"global"</i> you can specify training and inference settings for all tables. If the same settings are specified for a specific table, they will override the global settings</li>
<li>If the information about <i>"destination"</i> isn't specified in <i>"infer_settings"</i>, by default the synthetic data will be stored locally in <i>".csv"</i> format</li>
</ul>

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

The train and inference components of <i>syngen</i> is available as public docker image:

<https://hub.docker.com/r/tdspora/syngen>

To run dockerized code (see parameters description in *Training* and *Inference* sections) for one table call:

```bash
docker pull tdspora/syngen
docker run --rm \
  --user $(id -u):$(id -g) \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --task=train \
  --table_name=TABLE_NAME \
  --source=./model_artifacts/YOUR_CSV_FILE.csv

docker run --rm \
  --user $(id -u):$(id -g) \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --task=infer \
  --table_name=TABLE_NAME
```

PATH_TO_LOCAL_FOLDER is an absolute path to the folder where your original csv is stored.

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call.

To run dockerized code by providing the metadata file simply call:

```bash
docker pull tdspora/syngen
docker run --rm \
  --user $(id -u):$(id -g) \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --task=train \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker run --rm \
  --user $(id -u):$(id -g) \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --task=infer \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

You can add any arguments listed in the corresponding sections for infer and training processes in the CLI call, however, they will be
overwritten by corresponding arguments in the metadata file.

#### UI web interface
You can access the streamlit UI web interface by running the following command after installing the library with the UI option:

```bash
pip install syngen[ui]
```
then create a python file and insert the code provided below into it:

```python
from syngen import streamlit_app


streamlit_app.start()
```

run the python file:

```bash
python your_file.py
```

You also can access the streamlit UI web interface by launching the container with the following command:

```bash
docker pull tdspora/syngen
docker run -p 8501:8501 tdspora/syngen --webui
```

The UI will be available at <http://localhost:8501>.

#### MLflow monitoring

Set the `MLFLOW_TRACKING_URI` environment variable to the desired MLflow tracking server, for instance:
http://localhost:5000/. You can also set the `MLFLOW_ARTIFACTS_DESTINATION` environment variable to your preferred path 
(including the cloud path), where the artifacts should be stored. Additionally, set the `MLFLOW_EXPERIMENT_NAME` 
environment variable to the name you prefer for the experiment. 
To get the system metrics, please set the `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` environment variable to `true`.
By default, the metrics are logged every 10 seconds, but the interval may be changed by setting the environment variable 
`MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` (for more detailed description look [here](https://mlflow.org/docs/latest/system-metrics/index.html))

When using Docker, ensure the environmental variables are set before running the container.

The provided environmental variables allow to track the training process, and the inference process, and store 
the artifacts in the desired location.
You can access the MLflow UI by navigating to the provided URL in your browser. If you store artifacts in remote storage,
ensure that all necessary credentials are provided before using Mlflow.

```bash
docker pull tdspora/syngen:latest
docker run --rm -it \
  --user $(id -u):$(id -g) \
  -e MLFLOW_TRACKING_URI='http://localhost:5000' \
  -e MLFLOW_ARTIFACTS_DESTINATION=MLFLOW_ARTIFACTS_DESTINATION \
  -e MLFLOW_EXPERIMENT_NAME=test_name \
  -e MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true \
  -e MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL 10 \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML

docker run --rm -it \
  --user $(id -u):$(id -g) \
  -e MLFLOW_TRACKING_URI='http://localhost:5000' \
  -e MLFLOW_ARTIFACTS_DESTINATION=MLFLOW_ARTIFACTS_DESTINATION \
  -e MLFLOW_EXPERIMENT_NAME=test_name \
  -e MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true \
  -e MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL 10 \
  -v PATH_TO_LOCAL_FOLDER:/src/model_artifacts tdspora/syngen \
  --metadata_path=./model_artifacts/PATH_TO_METADATA_YAML
```

## Syngen Installation Guide for MacOS ARM (M1/M2) with Python 3.10

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.10
- Homebrew (optional but recommended for managing dependencies)

### Installation Steps

1. **Upgrade pip**: Ensure you have the latest version of `pip`.

    ```sh
    pip install --upgrade pip
    ```

2. **Install Setuptools, Wheel, and Cython**: These packages are necessary for building and installing other dependencies.

    ```sh
    pip install setuptools wheel 'Cython<3'
    ```

3. **Install Fastavro**: Install a specific version of `fastavro` to avoid build issues.

    ```sh
    pip install --no-build-isolation fastavro==1.5.1
    ```

4. **Install Syngen**: Now, you can install the Syngen package.

    ```sh
    pip install syngen
    ```

5. **Install TensorFlow Metal**: This package leverages the GPU capabilities of M1/M2 chips for TensorFlow.

    ```sh
    pip install tensorflow-metal
    ```

#### From source (development)

Download repository from GitHub by cloning or zip file.
Then install it in editable mode.

```sh
    pip install -e .
```

### Additional Information

- **Homebrew**: If you do not have Homebrew installed, you can install it by running:

    ```sh
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

- **Python 3.10**: Ensure you have Python 3.10 installed. You can use pyenv to manage different Python versions:

    ```sh
    brew install pyenv
    pyenv install 3.10.0
    pyenv global 3.10.0
    ```

### Verifying Installation

To verify the installation, run the following command to check if Syngen is installed correctly:

```sh
python -c "import syngen; print(syngen.__version__)"
```

If the command prints the version of Syngen without errors, the installation was successful.

### Troubleshooting

If you encounter any issues during installation, consider the following steps:

- Ensure all dependencies are up-to-date.
- Check for any compatibility issues with other installed packages.
- Consult the Syngen [documentation](https://github.com/tdspora/syngen) or raise an issue on GitHub.

## Contribution

We welcome contributions from the community to help us improve and maintain our public GitHub repository. We appreciate any feedback, bug reports, or feature requests, and we encourage developers to submit fixes or new features using issues.

If you have found a bug or have a feature request, please submit an issue to our GitHub repository. Please provide as much detail as possible, including steps to reproduce the issue or a clear description of the feature request. Our team will review the issue and work with you to address any problems or discuss any potential new features.

If you would like to contribute a fix or a new feature, please submit a pull request to our GitHub repository. Please make sure your code follows our coding standards and best practices. Our team will review your pull request and work with you to ensure that it meets our standards and is ready for inclusion in our codebase.

We appreciate your contributions, and thank you for your interest in helping us maintain and improve our public GitHub repository.
