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
  
def dataDownloader(src,dest):
  downloadPath = Path("downloadedData/")/dest

  if(downloadPath.is_dir()):
    print(f"{downloadPath} directory already exists, skipping downloading procedure")
  else:
    print(f"{downloadPath} directory doesn't already exists, starting downloading procedure")
    downloadPath.mkdir(parents=True,exist_ok=True)
    target = Path(src).name
    with open(Path("downloadedData/")/target,"wb") as f:
      requested = requests.get(src)
      print(f"Downloading {target} from {src}")
      f.write(requested.content)
    
    with zipfile.ZipFile(Path("downloadedData/")/target,"r") as zipRef:
      print(f"Unzipping the data")
      zipRef.extractall(downloadPath)
      os.remove(Path("downloadedData/")/target)
  return downloadPath

def computeAccu(model,dataLoader,device):
  model.eval()
  with torch.no_grad():
    correctPred = 0
    totalPred = 0

    for features,targets in dataLoader:
      features = features.to(device)
      targets  = targets.to(device)

      logits   = model(features)
      _ , predLabel = torch.max(logits,1)
      totalPred += targets.size(0)
      correctPred += (predLabel == targets).sum()

  return correctPred.float()/totalPred * 100