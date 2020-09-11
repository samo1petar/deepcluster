FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

LABEL maintainer='petar.pavlovic@toptal.com'

RUN apt update

RUN mkdir -p /home/deepcluster
RUN mkdir -p /home/results

COPY . /home/deepcluster/

WORKDIR /home/deepcluster

RUN apt install -y libgl1-mesa-glx
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install faiss-cpu==1.6.3 --no-cache
RUN pip3 install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python
