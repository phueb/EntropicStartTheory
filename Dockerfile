# this Dockerfile builds a reproducible environment in which the code in this repository can be run.

FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.7 python3.7-distutils python3.7-dev \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.7 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pyitlib scipy scikit-learn==0.21.3 seaborn matplotlib PyYaml pandas celluloid pyprind torch==1.6 numpy==1.21.0 tokenizers==0.10.1
RUN pip3 install https://github.com/phueb/Ludwig/archive/v4.0.7.tar.gz
RUN pip3 install https://github.com/phueb/Preppy/archive/v3.1.1.tar.gz
RUN pip3 install https://github.com/phueb/CategoryEval/archive/v5.0.1.tar.gz
RUN pip3 install https://github.com/UIUCLearningLanguageLab/AOCHILDES/archive/v3.0.0.tar.gz
RUN pip3 install https://github.com/UIUCLearningLanguageLab/AONewsela/archive/v1.1.0.tar.gz