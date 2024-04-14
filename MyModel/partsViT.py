from torch import nn
import torch

class multiHeadSelfAttentionBlock(nn.Module):
  def __init__(self,embeddingDim=768,numHeads=12,attnDropOut=0):
    super().__init__()
    self.layerNorm = nn.LayerNorm(normalized_shape=embeddingDim)
    self.multiheadAttn = nn.MultiheadAttention(embed_dim=embeddingDim,num_heads=numHeads,dropout=attnDropOut,batch_first=True)

  def forward(self,x):
    layNorm = self.layerNorm(x)
    attnOutPut, _ = self.multiheadAttn(query=layNorm,key=layNorm,value=layNorm)
    return attnOutPut

class MLPBlock(nn.Module):
  def __init__(self,embeddingDim,hiddenLayer,dropOut=0.1):
    super().__init__()
    self.MLP = nn.Sequential(
        nn.LayerNorm(normalized_shape = embeddingDim),
        nn.Linear(embeddingDim, hiddenLayer),
        nn.GELU(),
        nn.Dropout(dropOut),
        nn.Linear(hiddenLayer,embeddingDim),
        nn.Dropout(dropOut)
    )

  def forward(self,x):
    return self.MLP(x)

class transformerEncoderBlock(nn.Module):
  def __init__(self, embeddingDim, hiddenLayer,numHeads,MLPdropOut,attnDropOut=0):
    super().__init__()
    self.MSABlock = multiHeadSelfAttentionBlock(embeddingDim,numHeads,attnDropOut)
    self.MLPBlock = MLPBlock(embeddingDim,hiddenLayer,MLPdropOut)

  def forward(self,x):
    x = self.MSABlock(x) + x
    x = self.MLPBlock(x) + x
    return x

class patchNPositionalEmbeddingMaker(nn.Module):
  def __init__(self,inChannels,outChannels,patchSize,imgSize):
    super().__init__()
    self.outChannels = outChannels

    # outChannels is the same as embeddingDim
    self.patchSize = patchSize
    self.numPatches = int(imgSize**2/patchSize**2)
    self.patchMaker = nn.Conv2d(inChannels,outChannels, kernel_size=patchSize,stride=patchSize,padding=0)
    self.flattener  = nn.Flatten(start_dim=2,end_dim=3)
    self.classEmbedding = nn.Parameter(torch.randn(1,1,self.outChannels),requires_grad=True)
    self.PositionalEmbedding = nn.Parameter(torch.randn(1,self.numPatches+1,self.outChannels), requires_grad=True)

  def forward(self,x):
    batchSize = x.shape[0]
    imgRes = x.shape[-1]
    if(imgRes % self.patchSize ==0):
      pass
    else:
      assert imgRes % self.patchSize ==0, 'Input size must be div by patchSize'
    x = self.patchMaker(x)
    x = self.flattener(x)
    x = x.permute(0,2,1)
    classToken = self.classEmbedding.expand(batchSize,-1,-1)
    x = torch.cat((classToken,x),dim=1)
    x = x + self.PositionalEmbedding
    # batchSize = x.shape[0]
    # embeddingDim = x.shape[-1]
    return x

