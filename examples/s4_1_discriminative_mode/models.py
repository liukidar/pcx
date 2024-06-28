from typing import Callable
# Core dependencies
import jax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn



class VGG5(pxc.EnergyModule):
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

        self.vodes = [
            pxc.Vode(shape) for _, shape in zip(range(len(self.feature_layers)), [
                (128, input_size//2, input_size//2),
                (256, input_size//4, input_size//4), 
                (512, input_size//8, input_size//8),
                (512, input_size//16, input_size//16)
            ])
        ] + [
            pxc.Vode((self.nm_classes.get(),), energy_fn=pxc.se_energy if se_flag else pxc.ce_energy)]
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, beta: float = 1.0):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x)
        
        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")
    
class VGG7(pxc.EnergyModule):
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

        self.vodes = [
            pxc.Vode(shape) for _, shape in zip(range(len(self.feature_layers)), [
                (128, input_size//2, input_size//2),
                (128, input_size//2, input_size//2),
                (256, input_size//4, input_size//4), 
                (256, input_size//4 - 2, input_size//4 - 2),
                (512, (input_size//4 - 2)//2, (input_size//4 - 2)//2),
                (512, (input_size//4 - 2)//2 - 2, (input_size//4 - 2)//2 - 2)
            ])
        ] + [
            pxc.Vode((self.nm_classes.get(),), energy_fn=pxc.se_energy if se_flag else pxc.ce_energy)]
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, beta: float = 1.0):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")
    

class AlexNet(pxc.EnergyModule):
    def __init__(self, nm_classes: int, input_size, act_fn: Callable[[jax.Array], jax.Array], se_flag: bool) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)

        # Note we use a custom activation function and not exclusively ReLU since
        # it does not seem to perform as well as in backpropagation
        self.act_fn = px.static(act_fn)

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

        # self.long_skip_connection = pxnn.Linear(4096, self.nm_classes.get())

        # We define the Vode modules. Note that currently each vode requires its shape
        # to be manually specified. This will be improved in the near future as lazy
        # initialisation should be possible.
        self.vodes = [
                pxc.Vode(shape) for _, shape in zip(
                    range(len(self.feature_layers)), 
                    [(64, input_size//4, input_size//4), 
                     (192, input_size//8, input_size//8), 
                     (384, input_size//8, input_size//8), 
                     (256, input_size//8, input_size//8), 
                     (256, input_size//16, input_size//16)]
                )
            ] + [
                pxc.Vode((4096,)) for _ in range(len(self.classifier_layers) - 1)
            ] + [
                pxc.Vode((self.nm_classes.get(),), energy_fn=pxc.se_energy if se_flag else pxc.ce_energy)
            ]

        # Remember 'frozen' is a user specified attribute used later in the gradient function
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, beta: float = 1.0):
        # Nothing new here: we just define the forward pass of the network by iterating
        # through the blocks and vodes. Each block is followed by a vode, to split the
        # computation in indpendent chunks.
        for i, (block, node) in enumerate(zip(self.feature_layers, self.vodes[: len(self.feature_layers)])):
            for layer in block:
                x = layer(x)
            x = node(x)

        x = x.flatten()
        for i, (block, node) in enumerate(zip(self.classifier_layers, self.vodes[len(self.feature_layers):])):
            for layer in block:
                x = layer(x)

            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") + beta * (y - self.vodes[-1].get("u")))

        return self.vodes[-1].get("u")
    
class LinearModel(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nm_classes: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        self.nm_classes = px.static(nm_classes)
        
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, nm_classes)]

        # the default ruleset for a Vode is: `{"STATUS.INIT": ("h, u <- u",),}` which means:
        # "if the status is set to 'STATUS.INIT', everytime I set 'u', save that value not only
        # in 'u', but also in 'x', which is exactly the behvaiour of a forward pass.
        # By default if not specified, the behaviour is '* <- *', i.e., save everything passed
        # to the vode via __call__ (remember vode(a) equals to vode.set("u", a)).
        #
        # Since we are doing classification, we replace the last energy with the equivalent of
        # cross entropy loss for predictive coding.
        self.vodes = [
            pxc.Vode((hidden_dim,)) for _ in range(nm_layers - 1)
        ] + [pxc.Vode((nm_classes,), pxc.se_energy if se_flag else pxc.ce_energy)]
        
        # 'frozen' is not a magic word, we define it here and use it later to distinguish between
        # vodes we want to differentiate or not.
        # NOTE: any attribute of a Param (except its value) is treated automatically as static,
        # no need to specify it (but it's possible if you like more consistency,
        # i.e., ...frozen = px.static(True)).
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y, beta=1.0):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            # remember 'x = v(a)' corresponds to v.set("u", a); x = v.get("x")
            #
            # note that 'self.act_fn' is a StaticParam, so to access it we would have to do
            # self.act_fn.get()(...), however, all standard methods such as __call__ and
            # __getitem__ are overloaded such that 'self.act_fn.__***__' becomes
            # 'self.act_fn.get().__***__'
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")

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