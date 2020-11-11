import numpy as np
import MultipleEVM
import h5py
import torch
import time


data_path = './feature_train_efficient_b3_center_loss_fp16.npy'

N_classes = 413

tailsize=33998
cover_threshold=0.7
distance_multiplier=0.45
distance_function='cosine'
device='cuda'

output_path = f'./EVM_{distance_function}_model_imagenet_b3_tail{inte(tailsize)}_ct{int(100*cover_threshold)}_dm{int(100*distance_multiplier)}.hdf5'




startTime = time.time()

#Load features and labels
known = np.load(data_path,allow_pickle=True)
known = known[~np.isnan(known).any(axis=1)]
labels = []
features = []
for i in range(len(known)):
    labels.append(int(known[i][0]))
    features.append(known[i][1:])

classes = [[] for i in range(N_classes)]

print('\nAppending features to their respective class.')
#Get features of every class
for i,j in enumerate(labels):
    classes[j].append(features[i])
    
for class_num in range(len(classes)):
    classes[class_num] = torch.from_numpy(np.stack(classes[class_num]))

print('\nTime taken to load/prepare data *****************')
print(time.time() - startTime)
    
print('\nCreating an object of the EVM class & training the model.')
#Create an object of the EVM class, tailsize is the same as EVM
mevm = MultipleEVM.MultipleEVM(tailsize=tailsize, cover_threshold=cover_threshold, distance_multiplier=distance_multiplier, distance_function=distance_function, device=device)

#Train the model
mevm.train(classes,labels = list(range(0,len(classes))))

print('\nTime taken to train *****************')
print(time.time() - startTime)

print('\nSaving the model.')
#Save the model (uncomment whichever model to save)
mevm.save(output_path)

print('\nOverall runtime *****************')
print(time.time() - startTime)
