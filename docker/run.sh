#!/bin/bash

set -ex

docker image build -t pcax:latest -f ./Dockerfile ..
docker run --gpus all -it pcax:latest /bin/bash
