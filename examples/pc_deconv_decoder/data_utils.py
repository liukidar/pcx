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


def reconstruct_image(image_ids: list[int], predictor, dataset):
    os.makedirs("images", exist_ok=True)
    images = [dataset["img"][i] for i in image_ids]
    imgs = np.asarray(images, dtype=np.uint8)
    input = imgs.astype(np.float32) / 255.0
    preds = predictor(input)
    preds = np.clip(preds, 0.0, 1.0)
    pred_imgs = np.asarray(preds * 255.0, dtype=np.uint8)
    sep = np.zeros((imgs.shape[1], 3, imgs.shape[3]), dtype=np.uint8)
    for image_id, orig_img, pred_img in zip(image_ids, imgs, pred_imgs):
        two_images = np.concatenate((orig_img, sep, pred_img), axis=1)
        image = Image.fromarray(two_images)
        image.save(f"images/image_{image_id}.png")
