FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app/mtgradient

ENV PATH=$PATH:$HOME/.local/bin/

RUN DEBIAN_FRONTEND=noninteractive apt update && apt-get install -y --no-install-recommends tzdata

RUN apt install -y --no-install-recommends software-properties-common wget libxrender1 libxext6 libsm6 git curl build-essential \
    && add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && apt install -y python3.9 python3.9-dev python3.9-distutils \ 
    && ln -s /usr/bin/python3.9 /usr/bin/python \ 
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

RUN python -m pip install poetry

COPY poetry.lock pyproject.toml /app/mtgradient/

RUN python -m poetry config virtualenvs.create false && python -m poetry install --no-interaction
