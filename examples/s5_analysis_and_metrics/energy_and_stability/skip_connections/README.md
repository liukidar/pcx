# VGG19 Skip Connections Experiments

This repository contains experiments for evaluating the performance of VGG19 architecture with and without skip connections on the CIFAR-10 dataset. The experiments are divided into three main simulations:

1. **Best Hyperparameters Simulation**
2. **Hyperparameter Search Simulation**
3. **Three Seeds Simulation**

Additionally, a Jupyter notebook is provided for visualizing the results.

## Best Hyperparameters Simulation

The simulations for the best hyperparameters are hardcoded and can be run directly. The scripts used for these simulations are:

- `vgg19_noskip_bestparams.py`: Runs VGG19 without skip connections using the best hyperparameters.
- `vgg19_skip_bestparams.py`: Runs VGG19 with skip connections using the best hyperparameters.

## Hyperparameter Search Simulation

To perform a hyperparameter search, you will need to use the `stune` library. Follow the instructions in the [stune GitHub repository](https://github.com/liukidar/stune/tree/master) to install and configure `stune`.

### Running the Hyperparameter Search

1. **Install `stune`**: Follow the instructions from the `stune` GitHub page to install the library.
2. **Run the search**: Use the following commands to run the hyperparameter search for both configurations:

For VGG19 without skip connections:
```bash
python -m stune --exe VGG --study 1 --gpus 0,1,2,3,4,5,6,7 --n_trials 128:8 --tuner ssh --sampler grid --config VGG_noskip.yaml
```

For VGG19 with skip connections:
```bash
python -m stune --exe VGG --study 2 --gpus 0,1,2,3,4,5,6,7 --n_trials 128:8 --tuner ssh --sampler grid --config VGG_skip.yaml
```

Ensure you have the following YAML configuration files:
- `VGG_noskip.yaml`
- `VGG_skip.yaml`

## Three Seeds Simulation

To showcase the difference in performance over multiple seeds, use the following scripts:

- `vgg19_noskip_3seeds.py`: Runs VGG19 without skip connections using the same hyperparameters over three different seeds.
- `vgg19_skip_3seeds.py`: Runs VGG19 with skip connections using the same hyperparameters over three different seeds.

These simulations require `data_utils.py` for data handling.

## Visualizing Results

Use the Jupyter notebook `VGG.ipynb` to visualize the results from the best hyperparameter comparison and the three seeds simulation over 30 epochs.

## File Structure

The directory structure for this project is as follows:

```
skip_connections/
├── results_data/
│   ├── result_noskip.json
│   ├── result_skip.json
│   ├── skip_vs_noskip.pdf
│   └── skip_vs_noskip.png
├── results_seeds/
│   ├── skip_vs_noskip.pdf
│   ├── vgg_noskip_final4_0.json
│   ├── vgg_noskip_final4_1.json
│   ├── vgg_noskip_final4_2.json
│   ├── vgg_skip_final4_0.json
│   ├── vgg_skip_final4_1.json
│   ├── vgg_skip_final4_2.json
├── readme_skip.md
├── VGG.ipynb
├── VGG.yaml
├── vgg19_noskip_3seeds.py
├── vgg19_noskip_bestparams.py
├── vgg19_noskip_hyperparams.py
├── vgg19_skip_3seeds.py
├── vgg19_skip_bestparams.py
└── vgg19_skip_hyperparams.py
```

## Running the Scripts

To run the scripts, use the following commands:

For best hyperparameters:
```bash
python vgg19_noskip_bestparams.py
python vgg19_skip_bestparams.py
```

For three seeds simulation:
```bash
python vgg19_noskip_3seeds.py
python vgg19_skip_3seeds.py
```

Ensure you have the required dependencies installed and the appropriate configuration files in place.
