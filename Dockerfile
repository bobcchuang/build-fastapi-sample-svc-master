#FROM openaiacr.azurecr.io/auo/python36cv:0.0.3-x86
FROM --platform=$TARGETARCH python:3.6.9-slim AS main

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends wget unzip libgl1 libglib2.0-0 libgomp1 libegl1

# python packages
RUN --mount=type=bind,source=/wheel,target=/mnt/pypi \
    --mount=type=bind,source=/requirements.txt,target=/mnt/requirements.txt \
    pip3 install --no-cache-dir -r /mnt/requirements.txt 
    
WORKDIR /
ENV PORT 5001

RUN mkdir -p /app/static
COPY static/* /app/static/


WORKDIR /app

COPY main.py /app/main.py

COPY download_weight.sh /app/download_weight.sh
RUN /bin/sh download_weight.sh

CMD uvicorn main:app --host=0.0.0.0 --port=${PORT}
