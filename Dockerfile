FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app/mtgradient

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERISON=1.4.2
ENV PATH="$PATH:/root/.cache/pypoetry/virtualenvs/mtgradient-_H8ARtCb-py3.9/bin/: $HOME/.local/bin/:$POETRY_HOME/bin"

RUN DEBIAN_FRONTEND=noninteractive apt update && apt-get install -y --no-install-recommends tzdata

RUN apt install -y --no-install-recommends software-properties-common wget libxrender1 libxext6 libsm6 git curl build-essential \
    && add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && apt install -y python3.9 python3.9-dev python3.9-distutils \ 
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

RUN curl -sSL https://install.python-poetry.org | python3.9 && python3.9 -m pip install llvmlite

COPY pyproject.toml /app/mtgradient/

# COPY poetry.lock pyproject.toml /app/mtgradient/

RUN poetry install 