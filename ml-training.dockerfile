# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

WORKDIR src

COPY src/ .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-m", "syngen.train"]
