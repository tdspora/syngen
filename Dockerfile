# syntax=docker/dockerfile:1

FROM python:3.9-bookworm

WORKDIR ml_workdir
WORKDIR data

COPY requirements.txt .
COPY requirements-streamlit.txt .

RUN apt-get update && \
    apt-get install -y \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-streamlit.txt

COPY ml_workdir/ .
COPY ml_workdir/syngen/streamlit_app/.streamlit syngen/.streamlit
COPY ml_workdir/syngen/streamlit_app/.streamlit/config.toml /root/.streamlit/config.toml
ENV PYTHONPATH "${PYTHONPATH}:/ml_workdir/syngen"
ENTRYPOINT ["python3", "-m", "start"]
