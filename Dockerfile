# Select the base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Select the working directory
WORKDIR /app

# Setup image: install system dependencies etc.
# RUN ...

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
