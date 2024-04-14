import time
import torch
import numpy 
from HelperFunctions.helper import computeAccu
def modelTrainer(model1, numEpochs, trainLoader,testLoader,valLoader, opt, device,scheduler=None,schedulerOn='validAcc'):
  startTime = time.time()
  miniBatchLoss = []
  trainAccLoss = []
  valAccLoss = []
  valPlotAccLoss =[]
  for e in range(numEpochs):
    model1.train()
    for batchIdx, (features,targets) in enumerate(trainLoader):
      features = features.to(device)
      targets  = targets.to(device)

      logits = model1(features)
      # _, predLabel = torch.max(logits,1)

      cost = torch.nn.functional.cross_entropy(logits,targets)
      opt.zero_grad()
      cost.backward()
      opt.step()
      miniBatchLoss.append(cost.item())
      if ( not (batchIdx%50) ):
        print('Epoch:%03d/%03d | Batch:%03d/%03d |  Cost:%.4f' %(e+1, numEpochs, batchIdx, len(trainLoader), cost.item()))
    with torch.no_grad():
      print('Epoch:%03d/%03d |' %(e+1, numEpochs))
      trainLoss = computeAccu(model1, trainLoader,device)
      valLoss   = computeAccu(model1, valLoader  ,device)
      valAccLoss.append(valLoss)
      trainAccLoss.append(trainLoss.cpu().numpy())
      
      print(f'Train Acc {trainLoss :.4f}%')
      print(f'Val Acc   {valLoss:.4f}%')
      print(f'Time Taken: {((time.time()-startTime)/60):.2f} min')
      if(scheduler is not None):
        if(schedulerOn == 'validAcc'):
          scheduler.step(valAccLoss[-1])
        elif(schedulerOn == 'miniBatchLoss'):
          scheduler.step(miniBatchLoss[-1])
        else:
          raise ValueError(f'invalid choice for SchedulerOn {schedulerOn}')
      # valAccLoss[-1] = valLoss.detach().numpy()
      valPlotAccLoss.append(valLoss.cpu().numpy())
#     break
  testLoss = computeAccu(model1, testLoader, device)
  print(f'Test Acc   {testLoss:.4f}%')
  print(f'Total Time Taken: {((time.time()-startTime)/60):.2f} min')

  return miniBatchLoss, trainAccLoss, valPlotAccLoss 
