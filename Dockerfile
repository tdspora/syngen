# syntax=docker/dockerfile:1
#
# Base image: Docker Hardened Images (DHI) PyTorch. Runtime variants have no
# shell/package manager and run as the nonroot (65532) user, so all setup
# happens in the -dev build stage (root, shell, pip, build tools); only the
# venv and flattened source are copied into the shell-less runtime stage.
# RUN/ENTRYPOINT in the runtime stage must use exec form — there is no shell
# there to interpret shell-form strings.

## -----------------------------------------------------
## Build stage (dev variant: root, shell, pip, build tools)
FROM dhi.io/pytorch:2.11-cuda13.0-cudnn9-debian-dev AS build-stage

WORKDIR /src

# venv with --system-site-packages reuses the base image's preinstalled
# torch/CUDA stack instead of reinstalling/duplicating it.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies (and the package) from the consolidated pyproject.toml.
# The package is built from the src/ layout, then the sources are flattened
# into /src so the runtime keeps the historical layout: `python3 -m start`
# (the top-level src/start.py module) spawns `python syngen/train.py` from
# /src.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir . && \
    mv src/* . && \
    rm -rf src pyproject.toml README.md && \
    mkdir model_artifacts uploaded_files mlruns

## -----------------------------------------------------
## Runtime stage (non-dev variant: nonroot, no shell, no package manager)
FROM dhi.io/pytorch:2.11-cuda13.0-cudnn9-debian AS runtime-stage

ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp
ENV PATH="/opt/venv/bin:$PATH"
# CPU resource policy is selected at runtime by SYNGEN_DEPLOYMENT_MODE:
# dedicated (default) optimizes a single Syngen job; shared configures sleeping
# OpenMP/MKL wait threads for concurrent containers.
# /src lets `python -m start` and the `python syngen/train.py` subprocess it
# spawns resolve `import syngen`; /src/syngen lets that subprocess import the
# package's own top-level modules. (Base image does not define PYTHONPATH.)
ENV PYTHONPATH="/src:/src/syngen"
# torch 2.11's optimizer construction (Adam(foreach=True)) lazily imports
# torch._dynamo, whose cache_dir_utils calls getpass.getuser(); that raises
# KeyError when the container runs as an arbitrary UID with no matching
# /etc/passwd entry (e.g. an OpenShift-style arbitrary runAsUser) and no
# USER/LOGNAME env var set. Setting USER/LOGNAME avoids the pwd lookup.
ENV USER=nonroot
ENV LOGNAME=nonroot

WORKDIR /src

COPY --from=build-stage /opt/venv /opt/venv
COPY --from=build-stage --chown=65532:65532 /src /src

USER 65532
ENTRYPOINT ["python3", "-m", "start"]
