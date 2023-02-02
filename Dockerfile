# app/Dockerfile

FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/MannLabs/alphapeptstats.git .

RUN pip3 install -e.

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR "/app/alphastats/gui"

ENTRYPOINT ["streamlit", "run", "AlphaPeptStats.py", "--server.port=8501", "--server.address=0.0.0.0"]