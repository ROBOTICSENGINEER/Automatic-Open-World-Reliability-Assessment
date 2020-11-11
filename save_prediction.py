import numpy as np
import h5py
import cv2
import PIL 
from random import shuffle
import torch
from torchsummary import summary
from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time

from timm.models.efficientnet import efficientnet_b3  # timm library
from MultipleEVM import MultipleEVM
from torch.cuda.amp import autocast 

model_name = 'efficientnet_b3_imagenet_dm_45'
address_of_trained_model = './trained_efficientnet_b3_imagenet.pth.tar'
evm_model_path = '/scratch/mjafarzadeh/efficientb3_imagenet_EVM_model_tail33998_ct70_dm45_tensor.hdf5'



n_cpu = 32
batch_size = 256
image_size = 300
Number_of_Classes = 1000

tailsize = 33998
cover_threshold = 0.7
distance_multiplier = 0.55

np.random.seed(2)
torch.manual_seed(2)


class Average_Meter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0

  def update(self, val, n):
    if n > 0:
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class known_train_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_1000_train.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)


class known_val_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_1000_val.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)



class unknown_train_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_166.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)


image_transform_val = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

XY_train = known_train_data_class(transform = image_transform_val)
XY_val = known_val_data_class(transform = image_transform_val)
XY_unknown = unknown_train_data_class(transform = image_transform_val)

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
unknown_loader = DataLoader(dataset=XY_unknown, batch_size=batch_size, shuffle=False, num_workers=n_cpu)


model = efficientnet_b3(num_classes=Number_of_Classes)  #timm library

# loading trained model
state_dict_model = torch.load(address_of_trained_model)
#model.load_state_dict(state_dict_model)
from collections import OrderedDict
new_state_dict_model = OrderedDict()
for k, v in state_dict_model.items():
  name = k[7:] # remove `module.`
  new_state_dict_model[name] = v
model.load_state_dict(new_state_dict_model)


  
  
model.cuda()

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.to(device)


mevm = MultipleEVM(tailsize=tailsize, cover_threshold=cover_threshold, distance_multiplier=distance_multiplier)
mevm.load(evm_model_path)



torch.backends.cudnn.benchmark=True

#print(model)
summary(model, ( 3, image_size, image_size), batch_size=-1, device='cuda')



def foo_torch_to_numpy(y,Logit,Pr):
  softmax_scores = torch.nn.functional.softmax(Logit,dim=1)
  sm_value, sm_indices = torch.max(softmax_scores, 1)
  evm_value = np.amax(Pr, axis=1)
  evm_indices = np.argmax(Pr, axis=1)
  z = np.hstack((y.cpu().data.numpy()[:, None], sm_indices.cpu().data.numpy()[:, None], 
                 sm_value.cpu().data.numpy()[:, None], evm_indices[:, None], evm_value[:, None]))
  return z


top1_train = Average_Meter()
top5_train = Average_Meter()
top1_val = Average_Meter()
top5_val = Average_Meter()


top1_train.reset()
top5_train.reset()
top1_val.reset()
top5_val.reset()

N_train = XY_train.__len__()
N_val = XY_val.__len__()
N_unknown = XY_unknown.__len__()


# true label , softmax label, softmax value, emv label, evm value
prediction_train = np.empty([N_train, 5])
prediction_val = np.empty([N_val, 5])
prediction_unknown = np.empty([N_unknown, 5])

model.eval()
print("start extracting feature in train datasets")
t1 = time.time()
n = 0
with torch.no_grad():
  for i, (x,y) in enumerate(train_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    Pr = mevm.class_probabilities(FV)
    prediction_train[n:(n+x.size(0)), :] = foo_torch_to_numpy(y,Logit,Pr)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    top1_train.update(prec1.item(), x.size(0))
    top5_train.update(prec5.item(), x.size(0))
    n = n+x.size(0)
    
model.eval()    
print("start extracting feature in validation datasets")
n=0
with torch.no_grad():
  for i, (x,y) in enumerate(val_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    Pr = mevm.class_probabilities(FV)
    prediction_val[n:(n+x.size(0)), :] = foo_torch_to_numpy(y,Logit,Pr)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    top1_val.update(prec1.item(), x.size(0))
    top5_val.update(prec5.item(), x.size(0))
    n = n+x.size(0)

model.eval()    
print("start extracting feature in validation datasets")
n=0
with torch.no_grad():
  for i, (x,y) in enumerate(unknown_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    Pr = mevm.class_probabilities(FV)
    prediction_unknown[n:(n+x.size(0)), :] = foo_torch_to_numpy(y,Logit,Pr)
    n = n+x.size(0)


t2 = time.time()
print('train top1 accuracy', top1_train.avg)
print('train top5 accuracy ', top5_train.avg)
print('validation top1 accuracy', top1_val.avg)
print('validation top5 accuracy ', top5_val.avg)
print("epoch time = ", t2-t1)

np.save(file = ('./prediction_train_' + model_name + '.npy'), arr=prediction_train)
np.save(file = ('./prediction_val_' + model_name + '.npy'), arr=prediction_val)
np.save(file = ('./prediction_unknown_' + model_name + '.npy'), arr=prediction_unknown)

print('\nEnd')
