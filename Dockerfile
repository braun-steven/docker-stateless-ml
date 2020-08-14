# Select the base image
FROM nvcr.io/nvidia/pytorch:20.07-py3

# Select the working directory
WORKDIR /app

# Install SPFlow from the master branch
RUN git clone https://github.com/SPFlow/SPFlow && \
    cd SPFlow/src && \
    bash create_pip_dist.sh && \
    pip install dist/spflow-0.0.40-py3-none-any.whl

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
