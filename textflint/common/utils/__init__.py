import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .logger import logger
from .importing import LazyLoader
from .load import *
from .install import *
from .error import FlintError


