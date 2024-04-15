
import torch
import torchvision
from torch import nn
from helper import setAllSeeds
from ViT import ViT

def getViT(seed,classNames,DEVICE):
  setAllSeeds(seed)
  ViTModel = ViT(3,768,16,224,3072,12,0.1,12,len(classNames)).to(DEVICE)
  vitWeights = torchvision.models.ViT_B_16_Weights.DEFAULT
  vitTransforms = vitWeights.transforms()
  vit = torchvision.models.vit_b_16(weights=vitWeights).to(DEVICE)
  for param in vit.parameters():
    param.requires_grad = False
  vit.heads = nn.Linear(in_features=768, out_features=len(classNames)).to(DEVICE)
  return vit,vitTransforms
