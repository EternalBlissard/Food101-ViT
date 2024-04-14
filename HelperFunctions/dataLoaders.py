import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import random_split
import torch
def createDataLoadersFood(batchSize, numWorkers=0,validFraction =None, trainTransforms =None, testTransforms =None , splitSize=0.2,seed=42):
  if(trainTransforms is None):
    trainTransforms = transforms.ToTensor()
  if(testTransforms is None):
    testTransforms = transforms.ToTensor()

  trainDataset = datasets.Food101(root='data',
                              split="train",
                              transform=trainTransforms,
                              download=True)

  valDataset = datasets.Food101(root='data',
                              split="train",
                              transform=testTransforms)

  testDataset = datasets.Food101(root='data',
                              split="test",
                              transform=testTransforms)
  classNames = trainDataset.classes
  if(validFraction is not None):
    num = int(len(trainDataset) * splitSize)
    trainDataset, _ = random_split(trainDataset, lengths=[num, len(trainDataset)-num], generator=torch.manual_seed(seed))
    valDataset, _ = random_split(valDataset, lengths=[num, len(valDataset)-num], generator=torch.manual_seed(seed))
    trainDataset, _ = random_split(trainDataset, lengths=[num - int(num*validFraction), int(num*validFraction)], generator=torch.manual_seed(seed))
    _, valDataset = random_split(valDataset, lengths=[num - int(num*validFraction), int(num*validFraction)], generator=torch.manual_seed(seed))
    trainLoader = DataLoader(dataset=trainDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers,
                         drop_last = True)
    valLoader = DataLoader(dataset=valDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers)
  else:
    trainLoader = DataLoader(dataset=trainDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers,
                         drop_last = True,
                         shuffle=True)
  testDataset, _ = random_split(testDataset, lengths=[num, len(testDataset)-num], generator=torch.manual_seed(seed))
  testLoader = DataLoader(dataset=testDataset,
                         batch_size=batchSize,
                         shuffle=False,
                         num_workers=numWorkers)

  if(validFraction is None):
    return trainLoader,testLoader, classNames
  else:
    return trainLoader,valLoader,testLoader, classNames
  