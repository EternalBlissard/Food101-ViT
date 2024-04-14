import torch
from torch import nn
from MyModel.partsViT import patchNPositionalEmbeddingMaker,transformerEncoderBlock

class ViT(nn.Module):
  def __init__(self,inChannels,outChannels,patchSize,imgSize, hiddenLayer,numHeads,MLPdropOut,numTransformLayers,numClasses,embeddingDropOut=0.1,attnDropOut=0):
    super().__init__()
    self.EmbeddingMaker = patchNPositionalEmbeddingMaker(inChannels,outChannels,patchSize,imgSize)
    # self.transformerEncodingBlock = transformerEncoderBlock(outChannels,hiddenLayer,numHeads,MLPdropOut,attnDropOut)
    self.embeddingDrop = nn.Dropout(embeddingDropOut)
    self.TransformEncoder = nn.Sequential(*[transformerEncoderBlock(outChannels,hiddenLayer,numHeads,MLPdropOut,attnDropOut) for _ in range(numTransformLayers)])
    self.Classifier = nn.Sequential(nn.LayerNorm(normalized_shape=outChannels),
                                    nn.Linear(outChannels,numClasses))
  def forward(self,x):
    x = self.EmbeddingMaker(x)
    x = self.embeddingDrop(x)
    x = self.TransformEncoder(x)
    x = self.Classifier(x[:,0])
    return x
