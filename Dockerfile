# syntax=docker/dockerfile:1

FROM python:3.10-bookworm

WORKDIR src

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

COPY src/ .
COPY src/syngen/streamlit_app/.streamlit syngen/.streamlit
COPY src/syngen/streamlit_app/.streamlit/config.toml /root/.streamlit/config.toml
ENV PYTHONPATH "${PYTHONPATH}:/src/syngen"
ENTRYPOINT ["python3", "-m", "start"]
