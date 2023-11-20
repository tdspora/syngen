# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

WORKDIR src

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENTRYPOINT ["python3", "-m", "syngen.infer"]
