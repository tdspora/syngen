# syntax=docker/dockerfile:1

FROM python:3.11-slim-trixie

WORKDIR /src

# Install dependencies (and the package) from the consolidated pyproject.toml.
# The package is built from the src/ layout, then the sources are flattened into
# /src so the runtime keeps the historical layout: `python3 -m start` (the
# top-level src/start.py module) spawns `python syngen/train.py` from /src.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir . && \
    mv src/* . && \
    rm -rf src pyproject.toml README.md && \
    pip uninstall -y pip

ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp
# /src lets `python -m start` and the `python syngen/train.py` subprocess it
# spawns resolve `import syngen`; /src/syngen lets that subprocess import the
# package's own top-level modules. (Base image does not define PYTHONPATH.)
ENV PYTHONPATH="/src:/src/syngen"
RUN mkdir model_artifacts uploaded_files mlruns && \
    groupadd syngen && \
    useradd -mg syngen syngen && \
    chown -R syngen:syngen model_artifacts uploaded_files mlruns

USER syngen
ENTRYPOINT ["python3", "-m", "start"]
