# https://docs.streamlit.io/deploy/tutorials/docker
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY alphastats alphastats
COPY MANIFEST.in .
COPY pyproject.toml .
COPY README.md .
COPY requirements.txt .

RUN pip install .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["alphastats", "gui"]
