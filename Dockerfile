FROM ubuntu:20.04

SHELL [ "bin/bash", "-c" ]

EXPOSE 9000

RUN apt update &&\
    apt install -y python3-dev wget htop build-essential make &&\
    apt install -y python3-pip

COPY . /home/loan-defaults-detection/

RUN cd /home/loan-defaults-detection/ &&\
    pip3 install -r requirements.txt

CMD ["bash", "/home/loan-defaults-detection/pre_processing.sh", "&&", "bash", "/home/loan-defaults-detection/modeling.sh"]