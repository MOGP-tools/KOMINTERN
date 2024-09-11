import time
import pickle
import json
import zipfile

import numpy as np
np.random.seed(seed=12)
import pandas as pd

import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
import gpytorch as gp

from custom_profiler import profiler, magic_profiler #pip install git+https://github.com/KarGeekrie/customProfiler.git

def testHelloWorld():
    print("Hello")