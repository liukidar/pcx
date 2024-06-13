#!/bin/bash

# This is required to run the experiment in the docker container.

cp -r /home/pcax_tmp/* /home/pcax/
pip install -e /home/pcax
pip install hydra-core==1.3.2
python -c "import jax; print(jax.devices())"
cd /home/benchmark
echo "--------- SETUP COMPLETE ---------"
