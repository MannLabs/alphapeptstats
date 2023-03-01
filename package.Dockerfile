FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Install any necessary system dependencies here
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/MannLabs/alphapeptstats.git .
# Install Python packages and any other dependencies
RUN pip3 install -e.

# Copy the Python package files into the container
COPY alphapeptstats /app/alphapeptstats

# Set the entrypoint command to the package's setup.py file
ENTRYPOINT ["python", "-m", "setup.py"]