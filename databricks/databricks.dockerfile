# syntax=docker/dockerfile:1

# Build the initial docker image:
    FROM databricksruntime/standard:15.4-LTS AS builder

    # Set arguments to handle proper pip install comand due to syngen rc version present in requirements file
    # For local tests, use the following parameter to pass build argument:
    # --build-arg PIP_INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --use-pep517 --no-cache-dir -r requirements.txt"

    ARG PIP_INSTALL_CMD="pip install --use-pep517 --no-cache-dir -r requirements-databricks-15.4-LTS.txt"

    # Minimize the number of RUN commands and clean up cache and temporary files
    RUN apt-get update && \
        apt-get install -y gcc g++ ccache build-essential curl && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/{apt,dpkg,cache,log}
    COPY src /src
    COPY requirements-databricks-15.4-LTS.txt /requirements-databricks-15.4-LTS.txt
    RUN /databricks/python3/bin/${PIP_INSTALL_CMD}
    ENV MPLCONFIGDIR=/tmp
    ENV PYTHONPATH="${PYTHONPATH}:/src"
    WORKDIR /src
