from typing import Callable
# Core dependencies
import jax

# pcax
import pcax as px
import pcax.nn as pxnn

class VGG5(px.Module):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.se_flag = px.static(se_flag)

        self.feature_layers = [
            (
                pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        self.classifier_layers = [
            (
                pxnn.Linear(512 * (input_size//16) * (input_size//16), self.nm_classes.get()),
            ),
        ]

    def __call__(self, x: jax.Array):
        for block in self.feature_layers:
            for layer in block:
                x = layer(x)

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)

        return x
    
class VGG7(px.Module):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.se_flag = px.static(se_flag)

        self.feature_layers = [
            (
                pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            )
        ]
        self.classifier_layers = [
            (
                pxnn.Linear(512 * ((input_size//4 - 2)//2 - 2) * ((input_size//4 - 2)//2 - 2), self.nm_classes.get()),
            ),
        ]


    def __call__(self, x: jax.Array):
        for block in self.feature_layers:
            for layer in block:
                x = layer(x)

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)

        return x

class AlexNet(px.Module):
    def __init__(self, nm_classes: int, input_size, act_fn: Callable[[jax.Array], jax.Array], se_flag: bool) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)

        # Note we use a custom activation function and not exclusively ReLU since
        # it does not seem to perform as well as in backpropagation
        self.act_fn = px.static(act_fn)
        self.se_flag = px.static(se_flag)

        # We define the convolutional layers. We organise them in blocks just for clarity.
        # Ideally, pcax will soon support a "pxnn.Sequential" module to ease the definition
        # of such blocks. Layers are based on equinox.nn, so check their documentation for
        # more information.
        self.feature_layers = [
            (
                pxnn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2),
            ),
            (
                pxnn.Conv2d(64, 192, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2),
            ),
            (pxnn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)), self.act_fn),
            (pxnn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)), self.act_fn),
            (
                pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2),
            ),
        ]
        # We define the classifier layers. We organise them in blocks just for clarity.
        self.classifier_layers = [
            (pxnn.Linear(256 * (input_size//16) * (input_size//16), 4096), self.act_fn),
            (pxnn.Linear(4096, 4096), self.act_fn),
            (pxnn.Linear(4096, self.nm_classes.get()),),
        ]

    def __call__(self, x: jax.Array):
        for block in self.feature_layers:
            for layer in block:
                x = layer(x)

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)

        return x
    
class LinearModel(px.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nm_classes: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        self.nm_classes = px.static(nm_classes)
        self.se_flag = px.static(se_flag)
        
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, nm_classes)]

    def __call__(self, x: jax.Array):
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))

        x = self.layers[-1](x)

        return x

def get_model(
        model_name:str,
        nm_classes: int,
        act_fn: Callable[[jax.Array], jax.Array],
        input_size: int,
        se_flag: bool
):
    if model_name == "VGG5":
        return VGG5(nm_classes, input_size, act_fn, se_flag)
    elif model_name == "VGG7":
        return VGG7(nm_classes, input_size, act_fn, se_flag)
    elif model_name == "AlexNet":
        return AlexNet(nm_classes, input_size, act_fn, se_flag)
    elif model_name == "MLP":
        return LinearModel(input_dim=784, hidden_dim=128, nm_layers=4, nm_classes=10, se_flag=se_flag, act_fn=act_fn)
    else:
        raise ValueError(f"Unknown model name: {model_name}")