import numpy as np
import jax
from sklearn.datasets import make_moons, make_circles
from torchvision.datasets import MNIST, FashionMNIST
from PIL import Image
from tqdm import tqdm

import os
from omegaconf import DictConfig
import jax.numpy as jnp
from .cluster import ClusterManager

cluster = ClusterManager()


# Function to resize images ChatGPT-4
def resize_images(images: np.ndarray, new_w: int, new_h: int, method: str, verbose: bool):
    resized_images = []

    try:
        filter = getattr(Image.Resampling, method.upper())
    except AttributeError:
        raise ValueError(f"Unknown resampling method: {method}")

    for i in tqdm(range(images.shape[0]), desc="Resizing images", disable=not verbose):
        # Convert from [W, H] to [W, H, C]
        image = images[i]

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        # Resize image
        pil_image = pil_image.resize((new_w, new_h), filter)

        # Convert PIL Image back to numpy array
        resized_array = np.asarray(pil_image)

        # Append to the list of resized images
        resized_images.append(resized_array)

    return np.array(resized_images)


def process_torch_dataset(
    dataset_train,
    dataset_test,
    mean: float,
    std: float,
    batch_size: int,
    resize_enabled: bool,
    resize_size: int,
    resize_method: str,
    verbose: bool,
):
    # copy torch to numpy
    X = dataset_train.data.numpy()
    y = dataset_train.targets.numpy()
    X_test = dataset_test.data.numpy()
    y_test = dataset_test.targets.numpy()

    img_size = X.shape[-1]

    # if resize is enabled, resize the images
    if resize_enabled:
        X = resize_images(X, resize_size, resize_size, resize_method, verbose)
        X_test = resize_images(X_test, resize_size, resize_size, resize_method, verbose)
        img_size = resize_size

    # flatten for MLP
    X = X.reshape(-1, img_size * img_size)
    X_test = X_test.reshape(-1, img_size * img_size)

    # set input and output dimensions, given resize
    input_dim, output_dim = img_size * img_size, 10

    # build batches. drop last to ensure all batches are the same size
    X = X[: batch_size * (X.shape[0] // batch_size)]
    y = y[: batch_size * (y.shape[0] // batch_size)]
    X_test = X_test[: batch_size * (X_test.shape[0] // batch_size)]
    y_test = y_test[: batch_size * (y_test.shape[0] // batch_size)]

    assert X.shape[0] % batch_size == 0
    assert X_test.shape[0] % batch_size == 0
    assert X.shape[0] == y.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # normalize inputs
    X, X_test = X / 255.0, X_test / 255.0
    X, X_test = (X - mean) / std, (X_test - mean) / std

    return (X, y), (X_test, y_test), input_dim, output_dim


class FastDataloader:
    def __init__(self, X, y, batch_size, drop_last=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = len(X)
        self.num_batches = self.num_samples // self.batch_size


def get_data(
    dataset: str,
    num_samples: int,
    batch_size: int,
    num_classes: int,
    num_epochs: int,
    resize_enabled: bool,
    resize_size: int,
    resize_method: str,
    verbose: bool = False,
    **kwargs,
):
    if dataset == "two_moons":
        X, y = make_moons(n_samples=batch_size * (num_samples // batch_size), noise=0.2, random_state=42)
        X_test, y_test = make_moons(n_samples=batch_size * (num_samples // batch_size) // 2, noise=0.2, random_state=0)
        input_dim, output_dim = 2, 2

    elif dataset == "two_circles":
        X, y = make_circles(n_samples=batch_size * (num_samples // batch_size), noise=0.02, random_state=42)
        X_test, y_test = make_circles(n_samples=batch_size * (num_samples // batch_size) // 2, noise=0.02, random_state=0)
        input_dim, output_dim = 2, 2

    elif dataset == "mnist":
        dataset_train = MNIST(root=os.path.join(cluster.data_dir, "MNIST"), download=True, train=True)
        dataset_test = MNIST(root=os.path.join(cluster.data_dir, "MNIST"), download=True, train=False)
        # normalize inputs:
        # mean, std = jnp.concat([X, X_test], axis=0).mean(), jnp.concat([X, X_test], axis=0).std()
        mean = 0.13062754273414612
        std = 0.30810779333114624
        (X, y), (X_test, y_test), input_dim, output_dim = process_torch_dataset(
            dataset_train,
            dataset_test,
            mean,
            std,
            batch_size,
            resize_enabled,
            resize_size,
            resize_method,
            verbose,
        )

    elif dataset == "fashion_mnist":
        dataset_train = FashionMNIST(root=os.path.join(cluster.data_dir, "FashionMNIST"), download=True, train=True)
        dataset_test = FashionMNIST(root=os.path.join(cluster.data_dir, "FashionMNIST"), download=True, train=False)

        # normalize inputs:
        # mean, std = jnp.concat([X, X_test], axis=0).mean(), jnp.concat([X, X_test], axis=0).std()
        mean = 0.28604063391685486
        std = 0.35302424454689024
        (X, y), (X_test, y_test), input_dim, output_dim = process_torch_dataset(
            dataset_train, dataset_test, mean, std, batch_size, resize_enabled, resize_size, resize_method, verbose
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if num_classes > 0:
        output_dim = num_classes
    X, y = jnp.array(X), jnp.array(y)
    X_test, y_test = jnp.array(X_test), jnp.array(y_test)

    y, y_test = jax.nn.one_hot(y, output_dim), jax.nn.one_hot(y_test, output_dim)

    train_dl = list(zip(X.reshape(-1, batch_size, input_dim), y.reshape(-1, batch_size, output_dim)))
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, input_dim), y_test.reshape(-1, batch_size, output_dim)))

    # num_epochs = num_epochs // (num_samples // batch_size) # TODO

    return (X, y), (X_test, y_test), train_dl, test_dl, input_dim, output_dim, num_epochs


def test_dataset(cfg: DictConfig):
    train_dataset, test_dataset, train_loader, test_loader = get_data(cfg)
    print(train_dataset)
    print(test_dataset)
    sample_data = next(iter(train_loader))
    print(sample_data[0].shape, sample_data[1].shape)


if __name__ == "__main__":
    # test two moons

    for ds in ["two_moons", "two_circles", "mnist"]:
        cfg = DictConfig(
            {
                "data": {
                    "dataset": ds,
                    "batch_size": 64,
                    "num_samples": 1000,
                }
            }
        )
        test_dataset(cfg)
