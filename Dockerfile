# syntax=docker/dockerfile:1

FROM python:3.11-bookworm

WORKDIR /src

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential python3.11-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y pip

COPY src/ .
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp
ENV PYTHONPATH="${PYTHONPATH}:/src/syngen"
RUN mkdir model_artifacts uploaded_files mlruns && \
    groupadd syngen && \
    useradd -mg syngen syngen && \
    chown -R syngen:syngen model_artifacts uploaded_files mlruns

USER syngen
ENTRYPOINT ["python3", "-m", "start"]
