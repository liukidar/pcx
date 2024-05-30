from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import random

import torch
import numpy as np
import jax.numpy as jnp
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class CelebAAdapter(torchvision.datasets.CelebA):
    """Unfortunately, CelebA was shortsightedly published on GDrive
    that has an always exhausted daily limit on the number of automated downloads.
    Thus, you will have to download the dataset manually from here:
    https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    You need files:
    - Img/img_align_celeba.zip
    - Anno/list_attr_celeba.txt
    - Anno/identity_CelebA.txt
    - Anno/list_bbox_celeba.txt
    - Anno/list_landmarks_align_celeba.txt
    - Eval/list_eval_partition.txt

    Create a folder ~/tmp/celeba/celeba (yes, twice) and put all the files mentioned above in that directory without any additional directories.
    Unzip img_align_celeba.zip in the same directory. You may delete img_align_celeba.zip now if you want to save space.
    """

    def __init__(self, *args, train: bool = True, transform=None, **kwargs):
        # CelebA image size is 178 width and 218 height. Quite strange, huh? Let's make it 200x200.
        transform = transforms.Compose(
            [
                # Crop 10 pixels from the top and 8 pixels from the bottom
                lambda x: x.crop((0, 10, 178, 210)),
                # Increase width to 200
                lambda x: x.resize((200, 200), Image.LANCZOS),
                transform if transform is not None else lambda x: x,
            ]
        )
        super().__init__(*args, split="train" if train else "test", transform=transform, **kwargs)


VISION_DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "celeba": CelebAAdapter,
    "fasion_mnist": torchvision.datasets.FashionMNIST,
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(0)
    random.seed(seed)


def numpy_collate(batch):
    """This is a simple collate function that stacks numpy arrays used to interface the PyTorch dataloader with JAX.
    In the future we hope to provide custom dataloaders that are independent of PyTorch.
    """
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# The dataloader assumes cuda is being used, as such it sets 'pin_memory = True' and
# 'prefetch_factor = 2'. Note that the batch size should be constant during training, so
# we set 'drop_last = True' to avoid having to deal with variable batch sizes.
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            seed_everything(worker_seed)
            if worker_init_fn is not None:
                worker_init_fn(worker_id)

        g = torch.Generator()
        g.manual_seed(0)

        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )


@dataclass
class VisionData:
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    image_restore: Callable[[np.array], np.array]


def get_vision_dataloaders(
    *, dataset_name: str, batch_size: int, should_normalize: bool = False
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset_name not in VISION_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in {VISION_DATASETS.keys()}")
    ds_cls = VISION_DATASETS[dataset_name]

    save_path = "~/tmp/" + dataset_name

    train_dataset = ds_cls(
        save_path,
        download=True,
        train=True,
    )

    norm_mean = np.zeros(3)
    norm_std = np.ones(3)

    if should_normalize:
        d = train_dataset.data / 255.0
        assert d.shape[3] == 3  # Channel last
        norm_mean = d.mean(axis=(0, 1, 2))
        norm_std = d.std(axis=(0, 1, 2))
        assert norm_mean.shape == (3,)
        assert norm_std.shape == (3,)

    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std) if should_normalize else lambda x: x,
            lambda x: x.numpy(),
        ]
    )

    def image_restore(x: jnp.ndarray) -> np.ndarray:
        # Channel first: (batch, channel, height, width)
        assert x.shape[1] == 3
        if should_normalize:
            x = x * norm_std[None, :, None, None] + norm_mean[None, :, None, None]
        x = x.clip(0.0, 1.0)
        x = x * 255.0
        return np.asarray(x, dtype=np.uint8)

    train_dataset = ds_cls(
        save_path,
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = ds_cls(
        save_path,
        transform=t,
        download=True,
        train=False,
    )

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return VisionData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        image_restore=image_restore,
    )


def reconstruct_image(image_ids: list[int], predictor, dataset, image_restore, output_dir: Path):
    # Channel first: (batch, channel, height, width)
    output_dir.mkdir(exist_ok=True, parents=True)
    input = np.stack([dataset[i][0] for i in image_ids])
    images = image_restore(input)
    preds = predictor(input)
    pred_imgs = image_restore(preds)
    sep_width = 3
    sep = np.zeros((images.shape[1], images.shape[2], sep_width), dtype=np.uint8)
    for image_id, orig_img, pred_img in zip(image_ids, images, pred_imgs):
        two_images = np.concatenate((orig_img, sep, pred_img), axis=2)
        image = Image.fromarray(two_images.transpose(1, 2, 0))
        image.save(output_dir / f"image_{image_id}.png")


if __name__ == "__main__":
    data = get_vision_dataloaders(dataset_name="cifar10", batch_size=32)
    print(data.train_dataset[0][0].shape)
    reconstruct_image([0, 1, 2], lambda x: x, data.test_dataset, data.image_restore, Path("test"))
