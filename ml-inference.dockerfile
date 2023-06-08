# syntax=docker/dockerfile:1

FROM ubuntu:20.04

# Update package manager (apt-get)
# and install (with the yes flag `-y`)
# Python and Pip
RUN apt-get update \
    && apt-get install -y build-essential checkinstall\
    && apt-get install -y wget \
    && apt-get -y upgrade
RUN cd /opt/ \
    && wget https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz \
    && tar xzf Python-3.8.9.tgz \
    && cd Python-3.8.9 \
    && apt-get install -y python3-pip \
    && pip install --upgrade pip==23.1.2;

WORKDIR src

COPY src/ .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-m", "syngen.infer"]
