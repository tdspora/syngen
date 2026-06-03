# syntax=docker/dockerfile:1

# Build the initial docker image:
    FROM databricksruntime/standard:15.4-LTS AS builder

    # Installs the consolidated syngen-databricks package from pyproject.toml.
    # The databricks-build workflow copies databricks/pyproject.toml to the repo
    # root before building this image, so `.` resolves the databricks variant.
    # For local tests against an RC published to TestPyPI, override the command:
    # --build-arg PIP_INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --use-pep517 --no-cache-dir ."

    ARG PIP_INSTALL_CMD="pip install --use-pep517 --no-cache-dir ."

    # Minimize the number of RUN commands and clean up cache and temporary files
    RUN apt-get update && \
        apt-get install -y gcc g++ ccache build-essential curl && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/{apt,dpkg,cache,log}
    WORKDIR /src
    COPY pyproject.toml README.md ./
    COPY src ./src
    RUN /databricks/python3/bin/${PIP_INSTALL_CMD}
    ENV MPLCONFIGDIR=/tmp
    # Base image does not define PYTHONPATH; point it at the copied src layout.
    ENV PYTHONPATH="/src/src"
