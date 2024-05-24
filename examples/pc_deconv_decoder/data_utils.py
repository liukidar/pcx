import os

from datasets import load_dataset
import numpy as np
from PIL import Image


def load_cifar10():
    dataset = load_dataset("cifar10")

    return dataset["train"], dataset["test"]


def preprocess_data(batch):
    """Normalize and preprocess the images."""
    images = np.array(batch["img"], dtype=np.float32) / 255.0
    labels = np.array(batch["label"], dtype=np.int32)
    return images, labels


def get_batches(dataset, batch_size):
    """Yield batches from the dataset."""
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        yield preprocess_data(batch)


def reconstruct_image(model, params, dataset, image_id: int):
    img = np.asarray(dataset["img"][image_id], dtype=np.uint8)
    input = img.astype(np.float32) / 255.0
    pred = model.apply({"params": params}, input)
    pred_img = np.asarray(pred * 255.0, dtype=np.uint8)
    sep = np.zeros((img.shape[0], 5, img.shape[2]), dtype=np.uint8)
    two_images = np.concatenate((img, sep, pred_img), axis=1)
    image = Image.fromarray(two_images)
    os.makedirs("images", exist_ok=True)
    image.save(f"images/image_{image_id}.png")
