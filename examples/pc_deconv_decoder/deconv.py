from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

# Load the CIFAR10 dataset
dataset = load_dataset("cifar10")

# Split into train and test sets
train_dataset = dataset["train"]
test_dataset = dataset["test"]


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


class ConvEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x (32, 32, 3)
        x = nn.Conv(features=16, kernel_size=(4, 4), strides=(2, 2))(x)  # (16, 16, 16)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)  # (8, 8, 32)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)  # (4, 4, 64)
        x = nn.relu(x)
        return x


class ConvDecoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x (4, 4, 64)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)  # (8, 8, 32)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(4, 4), strides=(2, 2))(x)  # (16, 16, 16)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2))(x)  # (32, 32, 3)
        return nn.sigmoid(x)


class ConvAutoEncoder(nn.Module):

    def setup(self) -> None:
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize the model
model = ConvAutoEncoder()


def compute_loss(params, batch):
    images, _ = batch
    pred = model.apply({"params": params}, images)
    loss = jnp.mean((pred - images) ** 2)
    return loss


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    # params = model.init(rng, jnp.ones([1, 8, 8, 3]))["params"]
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        return compute_loss(params, batch)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


# Training loop
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.001)
num_epochs = 15
batch_size = 64

for epoch in range(num_epochs):
    for batch in get_batches(train_dataset, batch_size):
        state = train_step(state, batch)
    print(f"Epoch {epoch + 1}, Loss: {compute_loss(state.params, preprocess_data(train_dataset[:batch_size]))}")


def evaluate_model(state, test_dataset):
    """Evaluate the model on the test set."""
    test_loss = 0.0
    num_batches = 0
    for batch in get_batches(test_dataset, batch_size):
        test_loss += compute_loss(state.params, batch)
        num_batches += 1
    return test_loss / num_batches


test_loss = evaluate_model(state, test_dataset)
print(f"Test Loss: {test_loss}")


def reconstruct_image(image_id: int):
    img = np.asarray(test_dataset["img"][image_id], dtype=np.uint8)
    input = img.astype(np.float32) / 255.0
    pred = model.apply({"params": state.params}, input)
    pred_img = np.asarray(pred * 255.0, dtype=np.uint8)
    sep = np.zeros((img.shape[0], 5, img.shape[2]), dtype=np.uint8)
    two_images = np.concatenate((img, sep, pred_img), axis=1)
    image = Image.fromarray(two_images)
    image.save(f"image_{image_id}.png")


for i in range(20):
    reconstruct_image(i)
