import os
import sys

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
for arg in sys.argv:
    if arg.startswith('--cuda='):
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=')[1]
    elif arg == '--xla_preallocate':
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

from .pc import *
from .core.filter import f

from .core.util import move


__all__ = ["f"]
