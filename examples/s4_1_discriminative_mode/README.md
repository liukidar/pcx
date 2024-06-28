# Section 4.1 Discriminative Mode

## Folder Structure

The `hps` folder contains the hyperparameter configurations used for the experiments reported in the results table of the Discriminative Mode section. It is organized as follows:

```
hps/
├── MLP/
│   ├── FashionMNIST/
│   │   ├── MLP_PC-CE_FashionMNIST.yaml
│   │   ├── MLP_BP-SE_FashionMNIST.yaml
│   │   └── ...
│   └── MNIST/
│       ├── MLP_PC-SE_MNIST.yaml
│       ├── MLP_BP-CE_MNIST.yaml
│       └── ...
├── VGG5/
│   ├── CIFAR10/
│   │   ├── VGG5_PC-PN_CIFAR10.yaml
│   │   ├── VGG5_iPC_CIFAR10.yaml
│   │   └── ...
│   ├── CIFAR100/
│   │   ├── VGG5_BP-CE_CIFAR100.yaml
│   │   ├── VGG5_PC-SE_CIFAR100.yaml
│   │   └── ...
│   └── TinyImageNet/
│       ├── VGG5_PC-CE_TinyImageNet.yaml
│       ├── VGG5_BP-SE_TinyImageNet.yaml
│       └── ...
└── VGG7/
    ├── CIFAR100/
    │   ├── VGG7_PC-NN_CIFAR100.yaml
    │   ├── VGG7_BP-SE_CIFAR100.yaml
    │   └── ...
    └── TinyImageNet/
        ├── VGG7_PC-SE_TinyImageNet.yaml
        ├── VGG7_iPC_TinyImageNet.yaml
        └── ...
```

## Running the Experiments

To run an experiment, use the following command:

```
python PC/BP/iPC.py path_to_the_yaml_file
```

Replace `PC/BP/iPC` with the algorithm you want to run (PC, BP, or iPC) and `path_to_the_yaml_file` with the path to the corresponding YAML configuration file in the `hps` folder.

For example, to run the PC algorithm with the Center Nudging on the VGG5 model with the CIFAR100 dataset, use:

```
python PC.py hps/VGG5/CIFAR100/VGG5_PC-CN_CIFAR100.yaml
```

