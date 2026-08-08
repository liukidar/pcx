#!/bin/bash

set -ex

docker image build -t pcx:latest -f ./Dockerfile ..
docker run --gpus all -it pcx:latest /bin/bash
