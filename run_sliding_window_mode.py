import random
import numpy as np
from scipy import stats
import torch


random.seed(2)
np.random.seed(2)


number_of_tests = 10000
number_of_points = 1001


#device = "cuda"
device = "cpu"

alpha = 0.01

ond_train = np.load('.ond_train_efficientnet_b3_fp16_imagenet_dm_045.npy')
ond_val = np.load('./ond_val_efficientnet_b3_fp16_imagenet_dm_045.npy')
ond_unknown = np.load('./ond_unknown_efficientnet_b3_fp16_imagenet_dm_045.npy')


ond_train = ond_train[~np.isnan(ond_train).any(axis=1)]
ond_val = ond_val[~np.isnan(ond_val).any(axis=1)]
ond_unknown = ond_unknown[~np.isnan(ond_unknown).any(axis=1)]



L_train =  ond_train[:,0]
sm_train =  ond_train[:,2]
p_train =  ond_train[:,4]
L_val =  ond_val[:,0]
sm_val =  ond_val[:,2]
p_val =  ond_val[:,4]
L_unknown =  ond_unknown[:,0]
sm_unknown =  ond_unknown[:,2]
p_unknown =  ond_unknown[:,4]


tests_sm = np.zeros((24,2200,number_of_tests))
tests_p = np.zeros((24,2200,number_of_tests))

for index_rho in range(24):
  
  rho = (index_rho+2)/100

  print("\n\nrho = ", rho)
  
  n_unknown_1 = 11
  n_known_1 = 1100 - n_unknown_1
  n_unknown_2 = int(1100*rho)
  n_known_2 = 1100 - n_unknown_2
  n_unknown = n_unknown_1 + n_unknown_2
  n_known = n_known_1 + n_known_2
  print((n_known_1,n_unknown_1,n_known_1+n_unknown_1),(n_known_2,n_unknown_2,n_known_2+n_unknown_2), (n_known,n_unknown,n_known+n_unknown))
  
  for test_ind in range(number_of_tests):
     
    rng_val = np.random.default_rng()
    val_indecies = rng_val.choice(len(L_val), size=n_known, replace=False)
    rng_unknown = np.random.default_rng()
    unknown_indecies = rng_unknown.choice(len(L_unknown), size=n_unknown, replace=False)
    
    i_known_1 = val_indecies[:n_known_1]
    i_unknown_1 = unknown_indecies[:n_unknown_1]
    i_known_2 = val_indecies[n_known_1:]
    i_unknown_2 = unknown_indecies[n_unknown_1:]

    sm_known_1 = sm_val[i_known_1]
    sm_unknown_1 = sm_unknown[i_unknown_1]
    sm_known_2 = sm_val[i_known_2]
    sm_unknown_2 = sm_unknown[i_unknown_2]    

    p_known_1 = p_val[i_known_1]
    p_unknown_1 = p_unknown[i_unknown_1]
    p_known_2 = p_val[i_known_2]
    p_unknown_2 = p_unknown[i_unknown_2]
    
    is_known_1 = np.zeros(1100)
    is_known_2 = np.zeros(1100)
    is_known_1[:n_known_1] = 1
    is_known_2[:n_known_2] = 1
    np.random.shuffle(is_known_1)
    np.random.shuffle(is_known_2)
    

    if is_known_2[0] == 1:
      iii = (np.nonzero(1-is_known_2))[0][0]
      is_known_2[0] = 0
      is_known_2[iii] = 1
    

    counter_all = 0
    counter_known = 0
    counter_unknown = 0
    for i in range(1100):
      if is_known_1[counter_all] == 1:
        tests_sm[ index_rho , i , test_ind] = sm_known_1[counter_known]
        tests_p[ index_rho , i , test_ind] = p_known_1[counter_known]
        counter_known = counter_known + 1
      else:
        tests_sm[ index_rho , i , test_ind] = sm_unknown_1[counter_unknown]
        tests_p[ index_rho , i , test_ind] = p_unknown_1[counter_unknown]
        counter_unknown = counter_unknown + 1
      counter_all = counter_all + 1
    
    counter_all = 0
    counter_known = 0
    counter_unknown = 0
    for i in range(1100,2200):
      if is_known_2[counter_all] == 1:
        tests_sm[ index_rho , i , test_ind] = sm_known_2[counter_known]
        tests_p[ index_rho , i , test_ind] = p_known_2[counter_known]
        counter_known = counter_known + 1
      else:
        tests_sm[ index_rho , i , test_ind] = sm_unknown_2[counter_unknown]
        tests_p[ index_rho , i , test_ind] = p_unknown_2[counter_unknown]
        counter_unknown = counter_unknown + 1
      counter_all = counter_all + 1



sigma_one_sm_train = np.sqrt(np.mean((sm_train-1)**2))
sigma_one_p_train = np.sqrt(np.mean((p_train-1)**2))




def KL_Gaussian(mu, sigma, m, s):
  kl = np.log(s/sigma) + ( ( (sigma**2) + ( (mu-m) **2) ) / ( 2 * (s**2) ) ) - 0.5
  return kl





with torch.no_grad():
  sigma_one_sm_train_cuda = torch.tensor(sigma_one_sm_train).double().to(device)
  sigma_one_p_train_cuda = torch.tensor(sigma_one_p_train).double().to(device)
  mu_one_cuda = torch.tensor(1.0).double().to(device)
  
  tensor_sm = torch.from_numpy(tests_sm).double().to(device)
  tensor_p = torch.from_numpy(tests_p).double().to(device)
  
  tensor_KL_sm = torch.zeros(((24,2100,number_of_tests))).double().to(device)
  tensor_KL_p = torch.zeros(((24,2100,number_of_tests))).double().to(device)
  
  for w2 in range(100,2200):
    w1 = w2 - 100
    mu_sm_w = torch.mean(tensor_sm[:,w1:w2,:] , dim=1)
    mu_p_w = torch.mean(tensor_p[:,w1:w2,:] , dim=1)
    sigma_sm_w = torch.std(tensor_sm[:,w1:w2,:] , dim=1)
    sigma_p_w = torch.std(tensor_p[:,w1:w2,:] , dim=1)
    
    if (sigma_sm_w == 0).any():
      print("\n\nbug")
      print(torch.sum(sigma_sm_w == 0))
      for i in range(24):
        for j in range(2000):
          if sigma_sm_w[i,j]==0:
            print((i,j), mu_sm_w[i,j])
            print(tensor_sm[i,w1:w2,j])
    
    tensor_KL_sm[:,w1,:] = KL_Gaussian(mu=mu_sm_w, sigma=sigma_sm_w, m = mu_one_cuda, s=sigma_one_sm_train_cuda)
    tensor_KL_p[:,w1,:] = KL_Gaussian(mu=mu_p_w, sigma=sigma_p_w, m = mu_one_cuda, s=sigma_one_p_train_cuda)

  
with torch.no_grad():
  tensor_d_sm = tensor_KL_sm - tensor_KL_sm[:,0,:].view(tensor_KL_sm.shape[0],1, tensor_KL_sm.shape[2]).repeat((1,tensor_KL_sm.shape[1],1))
  tensor_d_p = tensor_KL_p - tensor_KL_p[:,0,:].view(tensor_KL_p.shape[0],1, tensor_KL_p.shape[2]).repeat((1,tensor_KL_p.shape[1],1))
  
  tensor_acc_sm = torch.zeros(((24,2100,number_of_tests))).double().to(device)
  tensor_acc_p = torch.zeros(((24,2100,number_of_tests))).double().to(device)

  
  for w in range(1,2100):
    # tensor_acc_sm[:,w,:] = torch.clamp(tensor_d_sm[:,w,:] + tensor_acc_sm[:,w-1,:], min=0.0, max=1.0 )
    # tensor_acc_p[:,w,:] = torch.clamp(tensor_d_p[:,w,:] + tensor_acc_p[:,w-1,:], min=0.0, max=1.0 )
    tensor_acc_sm[:,w,:] = torch.clamp( (alpha * tensor_d_sm[:,w,:]) + tensor_acc_sm[:,w-1,:], min=0.0 )
    tensor_acc_p[:,w,:] = torch.clamp( (alpha * tensor_d_p[:,w,:]) + tensor_acc_p[:,w-1,:], min=0.0)

  
print("median KL_sm before = ", torch.median( tensor_KL_sm[:,:1000]).item())
print("median KL_sm after = ", torch.median( tensor_KL_sm[:,1000:]).item())
print("median KL_evm before = ", torch.median( tensor_KL_p[:,:1000]).item())
print("median KL_evm after = ", torch.median( tensor_KL_p[:,1000:]).item())
print("median ACC_sm before = ", torch.median( tensor_acc_sm[:,:1000]).item())
print("median ACC_sm after = ", torch.median( tensor_acc_sm[:,1000:]).item())
print("median ACC_evm before = ", torch.median( tensor_acc_p[:,:1000]).item())
print("median ACC_evm after = ", torch.median( tensor_acc_p[:,1000:]).item())


print("min KL_sm before = ", torch.min( tensor_KL_sm[:,:1000]).item())
print("min KL_evm before = ", torch.min( tensor_KL_p[:,:1000]).item())
print("min ACC_sm before = ", torch.min( tensor_acc_sm[:,:1000]).item())
print("min ACC_evm before = ", torch.min( tensor_acc_p[:,:1000]).item())

print("max KL_sm before = ", torch.max( tensor_KL_sm[:,:1000]).item())
print("max KL_evm before = ", torch.max( tensor_KL_p[:,:1000]).item())
print("max ACC_sm before = ", torch.max( tensor_acc_sm[:,:1000]).item())
print("max ACC_evm before = ", torch.max( tensor_acc_p[:,:1000]).item())

print("min KL_sm after = ", torch.min( tensor_KL_sm[:,1000:]).item())
print("min KL_evm after = ", torch.min( tensor_KL_p[:,1000:]).item())
print("min ACC_sm after = ", torch.min( tensor_acc_sm[:,1000:]).item())
print("min ACC_evm after = ", torch.min( tensor_acc_p[:,1000:]).item())


print("max KL_sm after = ", torch.max( tensor_KL_sm[:,1000:]).item())
print("max KL_evm after = ", torch.max( tensor_KL_p[:,1000:]).item())
print("max ACC_sm after = ", torch.max( tensor_acc_sm[:,1000:]).item())
print("max ACC_evm after = ", torch.max( tensor_acc_p[:,1000:]).item())



with torch.no_grad():
  threshold = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  total_accuracy = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  pre_accuracy = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  mid_accuracy = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  post_accuracy = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  failure = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  early = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  on_time = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  late = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  MAE = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  no_fail = torch.zeros((24,number_of_points,4) , dtype = torch.double)
  
  interval_KL_sm = torch.linspace(0.0, 1.5, steps = number_of_points , dtype = torch.double)
  interval_KL_evm = torch.linspace(0.0, 50.0, steps = number_of_points , dtype = torch.double)
  interval_ACC_sm = torch.linspace(0.0, 6.0, steps = number_of_points , dtype = torch.double)
  interval_ACC_evm  = torch.linspace(0.0, 340.0, steps = number_of_points , dtype = torch.double)
  
  #(24,2100,number_of_tests)

  for point in range(number_of_points):
    print(f"Thresholds {point} from {number_of_points}")
    
    
    thresh_KL_sm = interval_KL_sm[point]
    thresh_KL_evm = interval_KL_evm[point]
    thresh_ACC_sm = interval_ACC_sm[point]
    thresh_ACC_evm = interval_ACC_evm[point]
  
    prediction_KL_sm = ((tensor_KL_sm>thresh_KL_sm).detach().clone()).double()
    prediction_KL_evm = ((tensor_KL_p>thresh_KL_evm).detach().clone()).double()
    prediction_ACC_sm = ((tensor_acc_sm>thresh_ACC_sm).detach().clone()).double()
    prediction_ACC_evm = ((tensor_acc_p>thresh_ACC_evm).detach().clone()).double()
  
    for j in range(1,2100):
      #(24,2100,number_of_tests)
      prediction_KL_sm[:,j,:] = torch.max(prediction_KL_sm[:,j-1,:] , prediction_KL_sm[:,j,:])
      prediction_KL_evm[:,j,:] = torch.max(prediction_KL_evm[:,j-1,:] , prediction_KL_evm[:,j,:])
      prediction_ACC_sm[:,j,:] = torch.max(prediction_ACC_sm[:,j-1,:] , prediction_ACC_sm[:,j,:])
      prediction_ACC_evm[:,j,:] = torch.max(prediction_ACC_evm[:,j-1,:] , prediction_ACC_evm[:,j,:])
  
    #(24,2100,number_of_tests)
    correctness_KL_sm = torch.zeros((24,2100,number_of_tests))
    correctness_KL_evm = torch.zeros((24,2100,number_of_tests))
    correctness_ACC_sm = torch.zeros((24,2100,number_of_tests))
    correctness_ACC_evm = torch.zeros((24,2100,number_of_tests))
    correctness_KL_sm[:,:1000,:] = 1 - prediction_KL_sm[:,:1000,:]
    correctness_KL_evm[:,:1000,:] = 1 - prediction_KL_evm[:,:1000,:]
    correctness_ACC_sm[:,:1000,:] = 1 - prediction_ACC_sm[:,:1000,:]
    correctness_ACC_evm[:,:1000,:] = 1 - prediction_ACC_evm[:,:1000,:]  
    correctness_KL_sm[:,1100:,:] = prediction_KL_sm[:,1100:,:]
    correctness_KL_evm[:,1100:,:] = prediction_KL_evm[:,1100:,:]
    correctness_ACC_sm[:,1100:,:] = prediction_ACC_sm[:,1100:,:]
    correctness_ACC_evm[:,1100:,:] = prediction_ACC_evm[:,1100:,:]  
    correctness_KL_sm[:,1000:1100,:] = torch.max(prediction_KL_sm[:,1000:1100,:] , (  1 - prediction_KL_sm[:,1000:1100,:]   )    *  (correctness_KL_sm[:,999,:].view(-1,1,number_of_tests).repeat(1,100,1)))
    correctness_KL_evm[:,1000:1100,:] = torch.max(prediction_KL_evm[:,1000:1100,:] , (  1 - prediction_KL_evm[:,1000:1100,:]   ) * (correctness_KL_evm[:,999,:].view(-1,1,number_of_tests).repeat(1,100,1)))
    correctness_ACC_sm[:,1000:1100,:] = torch.max(prediction_ACC_sm[:,1000:1100,:] , (  1 - prediction_ACC_sm[:,1000:1100,:]   ) * (correctness_ACC_sm[:,999,:].view(-1,1,number_of_tests).repeat(1,100,1)))
    correctness_ACC_evm[:,1000:1100,:] = torch.max(prediction_ACC_evm[:,1000:1100,:] , (  1 - prediction_ACC_evm[:,1000:1100,:])  * (correctness_ACC_evm[:,999,:].view(-1,1,number_of_tests).repeat(1,100,1)))
    
    #(24,2100,number_of_tests)
    prediction_pre_KL_sm  = prediction_KL_sm[:,:1000,:]
    prediction_pre_KL_evm  = prediction_KL_evm[:,:1000,:]
    prediction_pre_ACC_sm  = prediction_ACC_sm[:,:1000,:]
    prediction_pre_ACC_evm  = prediction_ACC_evm[:,:1000,:]
    prediction_mid_KL_sm  = prediction_KL_sm[:,1000:1100,:]
    prediction_mid_KL_evm  = prediction_KL_evm[:,1000:1100,:]
    prediction_mid_ACC_sm  = prediction_ACC_sm[:,1000:1100,:]
    prediction_mid_ACC_evm  = prediction_ACC_evm[:,1000:1100,:]    
    prediction_post_KL_sm = prediction_KL_sm[:,1100:,:] 
    prediction_post_KL_evm = prediction_KL_evm[:,1100:,:] 
    prediction_post_ACC_sm = prediction_ACC_sm[:,1100:,:] 
    prediction_post_ACC_evm = prediction_ACC_evm[:,1100:,:] 
    correctness_pre_KL_sm  = correctness_KL_sm[:,:1000,:]
    correctness_pre_KL_evm  = correctness_KL_evm[:,:1000,:]
    correctness_pre_ACC_sm  = correctness_ACC_sm[:,:1000,:]
    correctness_pre_ACC_evm  = correctness_ACC_evm[:,:1000,:]
    correctness_mid_KL_sm  = correctness_KL_sm[:,1000:1100,:]
    correctness_mid_KL_evm  = correctness_KL_evm[:,1000:1100,:]
    correctness_mid_ACC_sm  = correctness_ACC_sm[:,1000:1100,:]
    correctness_mid_ACC_evm  = correctness_ACC_evm[:,1000:1100,:]    
    correctness_post_KL_sm = correctness_KL_sm[:,1100:,:] 
    correctness_post_KL_evm = correctness_KL_evm[:,1100:,:] 
    correctness_post_ACC_sm = correctness_ACC_sm[:,1100:,:] 
    correctness_post_ACC_evm = correctness_ACC_evm[:,1100:,:] 
    
    threshold[:,point,0] = thresh_KL_sm
    threshold[:,point,1] = thresh_KL_evm
    threshold[:,point,2] = thresh_ACC_sm
    threshold[:,point,3] = thresh_ACC_evm
  
    #(24,2100,number_of_tests)
    WindowsxTests = 2100*number_of_tests
    total_accuracy[:,point,0] = torch.sum(correctness_KL_sm, dim=(1,2)) / WindowsxTests
    total_accuracy[:,point,1] = torch.sum(correctness_KL_evm, dim=(1,2)) / WindowsxTests
    total_accuracy[:,point,2] = torch.sum(correctness_ACC_sm, dim=(1,2)) / WindowsxTests
    total_accuracy[:,point,3] = torch.sum(correctness_ACC_evm, dim=(1,2)) / WindowsxTests
  
    pre_WindowsxTests = 1000*number_of_tests
    pre_accuracy[:,point,0] = torch.sum(correctness_pre_KL_sm, dim=(1,2)) / pre_WindowsxTests
    pre_accuracy[:,point,1] = torch.sum(correctness_pre_KL_evm, dim=(1,2)) / pre_WindowsxTests
    pre_accuracy[:,point,2] = torch.sum(correctness_pre_ACC_sm, dim=(1,2)) / pre_WindowsxTests
    pre_accuracy[:,point,3] = torch.sum(correctness_pre_ACC_evm, dim=(1,2)) / pre_WindowsxTests
  
    mid_WindowsxTests = 100*number_of_tests
    mid_accuracy[:,point,0] = torch.sum(correctness_mid_KL_sm, dim=(1,2)) / mid_WindowsxTests
    mid_accuracy[:,point,1] = torch.sum(correctness_mid_KL_evm, dim=(1,2)) / mid_WindowsxTests
    mid_accuracy[:,point,2] = torch.sum(correctness_mid_ACC_sm, dim=(1,2)) / mid_WindowsxTests
    mid_accuracy[:,point,3] = torch.sum(correctness_mid_ACC_evm, dim=(1,2)) / mid_WindowsxTests  
  
  
    post_WindowsxTests = 1000*number_of_tests
    post_accuracy[:,point,0] = torch.sum(correctness_post_KL_sm, dim=(1,2)) / post_WindowsxTests
    post_accuracy[:,point,1] = torch.sum(correctness_post_KL_evm, dim=(1,2)) / post_WindowsxTests
    post_accuracy[:,point,2] = torch.sum(correctness_post_ACC_sm, dim=(1,2)) / post_WindowsxTests
    post_accuracy[:,point,3] = torch.sum(correctness_post_ACC_evm, dim=(1,2)) / post_WindowsxTests
  
    #(24,2100,number_of_tests)
    failure[:,point,0] =  torch.sum((1 - prediction_post_KL_sm[:,-1,:]), dim=-1) / number_of_tests
    failure[:,point,1] =  torch.sum((1 - prediction_post_KL_evm[:,-1,:]), dim=-1) / number_of_tests
    failure[:,point,2] =  torch.sum((1 - prediction_post_ACC_sm[:,-1,:]), dim=-1) / number_of_tests
    failure[:,point,3] =  torch.sum((1 - prediction_post_ACC_evm[:,-1,:]), dim=-1) / number_of_tests
  
  
    early[:,point,0] =  torch.sum(prediction_pre_KL_sm[:,-1,:], dim=-1)/ number_of_tests
    early[:,point,1] =  torch.sum(prediction_pre_KL_evm[:,-1,:], dim=-1)/ number_of_tests
    early[:,point,2] =  torch.sum(prediction_pre_ACC_sm[:,-1,:], dim=-1)/ number_of_tests
    early[:,point,3] =  torch.sum(prediction_pre_ACC_evm[:,-1,:], dim=-1)/ number_of_tests
  
    on_time[:,point,0] = torch.sum( ( (1 - prediction_pre_KL_sm[:,-1,:]) * prediction_post_KL_sm[:,0,:]) , dim=-1 ) / number_of_tests
    on_time[:,point,1] = torch.sum( ( (1 - prediction_pre_KL_evm[:,-1,:]) * prediction_post_KL_evm[:,0,:]) , dim=-1) / number_of_tests
    on_time[:,point,2] = torch.sum( ( (1 - prediction_pre_ACC_sm[:,-1,:]) * prediction_post_ACC_sm[:,0,:]) , dim=-1) / number_of_tests
    on_time[:,point,3] = torch.sum( ( (1 - prediction_pre_ACC_evm[:,-1,:]) * prediction_post_ACC_evm[:,0,:]) , dim=-1 ) / number_of_tests
  
    late[:,point,0] = torch.sum( ( (1 - prediction_post_KL_sm[:,0,:]) * prediction_post_KL_sm[:,-1,:]), dim=-1 ) / number_of_tests
    late[:,point,1] = torch.sum( ( (1 - prediction_post_KL_evm[:,0,:]) * prediction_post_KL_evm[:,-1,:]) , dim=-1 )/ number_of_tests
    late[:,point,2] = torch.sum( ( (1 - prediction_post_ACC_sm[:,0,:]) * prediction_post_ACC_sm[:,-1,:]) , dim=-1 ) / number_of_tests
    late[:,point,3] = torch.sum( ( (1 - prediction_post_ACC_evm[:,0,:]) * prediction_post_ACC_evm[:,-1,:]) , dim=-1 ) / number_of_tests
    
    
    #(24,2100,number_of_tests) ==> (24,number_of_tests)
    #pytorch 1.6.0 has a bug in torch.max
    location_KL_sm = torch.from_numpy(np.argmax(prediction_KL_sm.cpu().data.numpy(),axis=1)).double()
    location_KL_evm = torch.from_numpy(np.argmax(prediction_KL_evm.cpu().data.numpy(),axis=1)).double()
    location_ACC_sm = torch.from_numpy(np.argmax(prediction_ACC_sm.cpu().data.numpy(),axis=1)).double()
    location_ACC_evm = torch.from_numpy(np.argmax(prediction_ACC_evm.cpu().data.numpy(),axis=1)).double()

    # (24,number_of_tests) == > (24,number_of_tests)
    error_KL_sm = (1000.0 - location_KL_sm) * ((location_KL_sm<1000.0).double()) + (location_KL_sm - 1100.0) * ((location_KL_sm>1100).double())
    error_KL_evm = (1000.0 - location_KL_evm) * ((location_KL_evm<1000.0).double()) + (location_KL_evm - 1100.0) * ((location_KL_evm>1100).double())
    error_ACC_sm = (1000.0 - location_ACC_sm) * ((location_ACC_sm<1000.0).double()) + (location_ACC_sm - 1100.0) * ((location_ACC_sm>1100).double())
    error_ACC_evm = (1000.0 - location_ACC_evm) * ((location_ACC_evm<1000.0).double()) + (location_ACC_evm - 1100.0) * ((location_ACC_evm>1100).double())


    #(24,number_of_tests) ==> (24)
    MAE[:,point,0] = torch.mean( error_KL_sm , dim=1   )
    MAE[:,point,1] = torch.mean( error_KL_evm , dim=1   )
    MAE[:,point,2] = torch.mean( error_ACC_sm , dim=1   )
    MAE[:,point,3] = torch.mean( error_ACC_evm , dim=1   )
    
    
    #(24,2100,number_of_tests) ==> (24,number_of_tests)  ==> (24)
    no_fail[:,point,0] = torch.mean(  prediction_post_KL_sm[:,-1,:]  , dim=1   )
    no_fail[:,point,1] = torch.mean(  prediction_post_KL_evm[:,-1,:]  , dim=1   )
    no_fail[:,point,2] = torch.mean(  prediction_post_ACC_sm[:,-1,:]  , dim=1   )
    no_fail[:,point,3] = torch.mean(  prediction_post_ACC_evm[:,-1,:]  , dim=1   )




np.savez('sliding_window_array.npz', threshold=threshold.cpu().data.numpy(), 
         total_accuracy=total_accuracy.cpu().data.numpy(), pre_accuracy=pre_accuracy.cpu().data.numpy(),
         mid_accuracy=mid_accuracy.cpu().data.numpy(), post_accuracy=post_accuracy.cpu().data.numpy(), 
         failure=failure.cpu().data.numpy(), early=early.cpu().data.numpy(), on_time=on_time.cpu().data.numpy(), 
         late=late.cpu().data.numpy(), MAE=MAE.cpu().data.numpy(), no_fail=no_fail.cpu().data.numpy())

