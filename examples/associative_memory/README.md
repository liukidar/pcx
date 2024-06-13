# Using PCX for associative memory tasks

## Usage

This is where you could run associative memory tasks with generative PCNs with two simple steps:

- While you are in the root directory of `pcax`, `cd` to `examples/associative_memory`;
- Run `python associative_memory.py` to get things going.

There are 3 command line inputs you could control:

- `--corruption`: choose between `noise` and `mask` to train PCNs that retrieve memory from noise- or mask-corrupted images;
- `--hidden_dim`: choose within range $[512, 1024, 2048]$, which controls the number of hidden dimensions $d$ in a MLP generative PCN with shape $[512,d,d,12288]$ where $12288$ is the flattened Tiny ImageNet images' dimension;
- `--sample_size`: choose within range $[50, 100, 250, 500, 1000]$, which controls the number of images in memory. 

After running, a folder `results` will be created in the `associative_memory` folder that stores the recalled memories.

## Things to note

- **Hyperparameters**: In the `associative_memory` folder we also have a `hps.json` file that stores the searched hyperparameters, namely `w_lr` (parameter learning rate), `h_lr` (inference learning rate), `T_train` (number of inference steps in training), `T_gen` (number of inference steps in recall), that will give the best recall MSE in different settings. If you have correctly chosen the type of corruption, hidden dimenions and sample sizes from the specified ranges above, the code will automatically pick up the best hyperparameters from `hps.json`. However, if you choose some values outside of the ranges, you will get an error, in which case you will have to select or tune your own hyperparameters and change the values of `h_lr`, `w_lr`, `T_train` and `T_gen` accordingly in the code.
- **Datasets**: The code currently only supports associative memory tasks on Tiny ImageNet. It assumes that your Tiny ImageNet dataset is stored in `../../../datasets`. However, you can change this directory specification to wherever you like in the `get_dataloader` function, and the code will automatically download the Tiny ImagenNet dataset into your specified directory. You can also try out other datasets (ways to load other datasets can be found in other examples).