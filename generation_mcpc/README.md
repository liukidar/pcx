# MCPC with PCX
This folder can be used to run the MCPC experiments in PCX.

## Installation
To setup your environment for the experiments please follow the installation guidelines of the repo followed by installing the libraries specific to these experiments.
using:

```
pip install -r requirements.txt
```

## Usage
- (figure 2 left) To train an MCPC model on the iris dataset please run
```
python mcpc_iris.py
```
This code also shows how the non-linear model shown in the figure as well as a linear model can be fitted to the iris data.

- (figure 2 top-right + FID and Inceptions score of section 4.2) To train the MCPC model on the MNIST dataset in a fully unsupervised way you can run  
```
python mcpc_mnist.py --is-verbose true --metric fid             # to train an fid optimised model
python mcpc_mnist.py --is-verbose true --metric inception-score # train inception score optimised model
```
Additionally, you can set the argument `--seed seed_number` to set the training seed of the model. This code will print the FID and inception score every 10 epoch. 100 generated images will be saved after training along with a plot of how the fid, incpetion-score and model energy change during training.

- (figure 2 bottom-right) To train the MCPC model on the MNIST dataset for conditional gneeraiton based on the binary label even or odd you can run  
```
python mcpc_mnist_conditional.py --is-verbose true
```
This code has the same outputs as `mcpc_mnist.py`
- (VAE FID and Inceptions score of section 4.2) To train a VAE on the MNIST dataset in a fully unsupervised way you can run  
```
python vae_mnist.py --is-verbose true --metric fid             # to train an fid optimised model
python vae_mnist.py --is-verbose true --metric inception-score # train inception score optimised model
```
Additionally, you can set the argument `--seed seed_number` to set the training seed of the model. This code the same outputs as `mcpc_mnist.py`. It is also fully written in pytorch and should therefore be run in an python environment with pytorch-gpu for optimal training speed.


## Acknowledgements
This code is build upon multiple opensource code. We would like to aknoledge:
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid.git) that we modified to be used for the MNIST dataset. We modified the underlying classifier from InceptionV3 to an MNIST-trained ResNet18.
- [MNIST inception-score](https://github.com/sundyCoder/IS_MS_SS.git) that we modified so that it can be used by inputing a dolfer with MNIST images similarly to pytorch fid.
- [Pytorch MNIST VAE](https://github.com/lyeoni/pytorch-mnist-VAE.git) upon which we wrote `vae_mnist.py`.
