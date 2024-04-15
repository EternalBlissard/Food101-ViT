
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import zipfile
from pathlib import Path
import requests

def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  

