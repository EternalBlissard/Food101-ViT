import torch
from torch import nn
import torchvision
from torchvision import transforms
import sys
import matplotlib.pyplot as plt
from torchinfo import summary



from HelperFunctions.plottingHelper import plotTrainingLoss, plotAccuracy
from HelperFunctions.dataLoaders import createDataLoadersFood
from HelperFunctions.helper import setAllSeeds, dataDownloader
from HelperFunctions.trainerHelper import modelTrainer
from MyModel.ViT import ViT

print("cuda", torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#### Running param
batchSize = 32
numEpoch = 50
resultsDir = './'
######

trainTransforms = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.Resize(size=(256, 256), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop([224,224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
testTransforms = transforms.Compose([
    transforms.Resize(size=(256, 256), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop([224,224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
trainDataLoader,valDataLoader,testDataLoader, classNames = createDataLoadersFood(batchSize,validFraction=0.1,trainTransforms=trainTransforms,testTransforms=testTransforms, splitSize=0.4)

model = ViT(3,768,16,224,3072,12,0.1,12,len(classNames))
# with open('./model2/model.log', 'w') as f:
#     report = summary(model=model,input_size=(32,3,224,224),
#         col_names =['input_size','output_size','num_params','trainable'],
#         col_width=20,row_settings=['var_names'],device="cpu")
#     f.write(str(report))
# print(report)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.999),weight_decay=0.1)
schd = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       mode='max',
                                                       verbose=True)
setAllSeeds(42)
miniBatchLossList, trainAccList, validAccList = modelTrainer(
    model1=model,
    numEpochs=numEpoch,
    trainLoader=trainDataLoader,
    valLoader=valDataLoader,
    testLoader=testDataLoader,
    opt=optimizer,
    logBatch=100,
    saveBatch=3,
    device=DEVICE,
    scheduler=schd,
    schedulerOn='validAcc',
    gradIt=True,
    resultsDir=resultsDir)

plotTrainingLoss(miniBatchLoss=miniBatchLossList,
                   numEpoch=numEpoch,
                   iterPerEpoch=len(trainDataLoader),
                   avgIter=20,
                   resultsDir=resultsDir)    #plotTrainingLoss(miniBatchLoss,numEpoch,iterPerEpoch,resultsDir=None,avgIter = 100)
plt.show()
plotAccuracy(trainAccList,validAccList,resultsDir=resultsDir)
plt.ylim([0,100])
plt.show()
torch.save(model.state_dict(), resultsDir+'/finalModel.pt')
torch.save(optimizer.state_dict(), resultsDir+'/finalOptimizer.pt')
