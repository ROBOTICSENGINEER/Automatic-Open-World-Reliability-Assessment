import random
import numpy as np
from scipy import stats


random.seed(2)
np.random.seed(2)


number_of_tests = 10000
batch_size = 100
number_of_points = 1001

Delta_old = 0.4
Delta = 0.5
rho_hat = 0.01

ond_train = np.load('./ond_train_efficientnet_b3_fp16_imagenet_dm_045.npy')
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


def Kullback_Leibler(mu, sigma, m, s):
  kl = np.log(s/sigma) + ( ( (sigma**2) + ( (mu-m) **2) ) / ( 2 * (s**2) ) ) - 0.5
  return kl



def Bi_Kullback_Leibler(mu_1, mu_2, sigma_1, sigma_2, rho, m_1, m_2, s_1, s_2, r):
  # P ==>  mu_1, mu_2, sigma_1, sigma_2, rho
  # Q ==> m1, m2, s1, s2, r
  D1 = 2 * (1 - (r**2))
  N1 = ( (mu_1 - m_1) ** 2  ) / ( s_1 **2 )
  N2 = - ( 2 * r * (mu_1 - m_1) * (mu_2 - m_2) ) / ( s_1 * s_2 )
  N3 = ( (mu_2 - m_2) ** 2  ) / (s_2 **2 )
  N4 = ( (sigma_1 - s_1) ** 2  ) / ( s_1 **2 )
  N5_1 = rho * sigma_1 * sigma_2
  N5_2 = r * s_1 * s_2
  N5 = - ( 2 * r * ( N5_1 - N5_2 ) ) / ( s_1 * s_2 )
  N6 = ( (sigma_2 - s_2) ** 2  ) / (s_2 **2 )
  N7 = s_1 * s_2 * np.sqrt( 1 - (r**2)  )
  D7 = sigma_1 * sigma_2 * np.sqrt( 1 - (rho**2)  )
  kl_1 = (N1 + N2 + N3 + N4 + N5 + N6) / D1
  kl_2 = np.log( N7 / D7 )
  kl = kl_1 + kl_2
  return kl



mu_sm_train = np.mean(sm_train)
mu_evm_train = np.mean(p_train)
sigma_sm_train = np.std(sm_train)
sigma_evm_train = np.std(p_train)
corr_train = stats.pearsonr(sm_train, p_train)[0]
sigma_one_sm_train = np.sqrt(np.mean((sm_train-1)**2))
sigma_one_evm_train = np.sqrt(np.mean((p_train-1)**2))


threshold = np.zeros((24,number_of_points,7))
total_accuracy = np.zeros((24,number_of_points,7))
pre_accuracy = np.zeros((24,number_of_points,7))
post_accuracy = np.zeros((24,number_of_points,7))
failure = np.zeros((24,number_of_points,7))
early = np.zeros((24,number_of_points,7))
on_time = np.zeros((24,number_of_points,7))
late = np.zeros((24,number_of_points,7))
mean_early = np.zeros((24,number_of_points,7))
median_early = np.zeros((24,number_of_points,7))
maximum_early = np.zeros((24,number_of_points,7))
mean_late = np.zeros((24,number_of_points,7))
median_late = np.zeros((24,number_of_points,7))
maximum_late = np.zeros((24,number_of_points,7))
absolute_error = np.zeros((24,number_of_points,7))



for index_rho in range(24):
  
  rho = (index_rho+2)/100
  
  print("\n\nrho = ", rho)
  
  
  varepsilon = np.zeros((number_of_tests,40))
  EV_sm = np.zeros((number_of_tests,40))
  KL_sm  = np.zeros((number_of_tests,40))
  KL_evm  = np.zeros((number_of_tests,40))
  KL_short  = np.zeros((number_of_tests,40))
  KL_full  = np.zeros((number_of_tests,40))
  old_ond = np.zeros((number_of_tests,40)) 
  
  n_unknown_1 = 20
  n_unknown_2 = int(2000*rho)
  n_known_1 = 2000 - n_unknown_1
  n_known_2 = 2000 - n_unknown_2
  n_unknown = n_unknown_1 + n_unknown_2
  n_known = n_known_1 + n_known_2
  
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
    
    is_known_1 = np.zeros(2000)
    is_known_2 = np.zeros(2000)
    is_known_1[:n_known_1] = 1
    is_known_2[:n_known_2] = 1
    np.random.shuffle(is_known_1)
    np.random.shuffle(is_known_2)
    
    counter_all = 0
    counter_known = 0
    counter_unknown = 0
    ACC = 0.0
    for epoch in range(20):
      sm_epoch = np.zeros(batch_size)
      p_epoch = np.zeros(batch_size)
      for i in range(batch_size):
        if is_known_1[counter_all] == 1:
          sm_epoch[i] = sm_known_1[counter_known]
          p_epoch[i] = p_known_1[counter_known]
          counter_known = counter_known + 1
        else:
          sm_epoch[i] = sm_unknown_1[counter_unknown]
          p_epoch[i] = p_unknown_1[counter_unknown]
          counter_unknown = counter_unknown + 1
        counter_all = counter_all + 1
      
      mu_1 = np.mean(sm_epoch)
      mu_2 = np.mean(p_epoch)
      sigma_1 = np.std(sm_epoch)
      sigma_2 = np.std(p_epoch)
      rho_epoch = stats.pearsonr(sm_epoch, p_epoch)[0]
      
      v_epoch = 1 - Delta - p_epoch
      nu_epoch = np.maximum(v_epoch,np.zeros(batch_size))
      #print(f"{epoch=}\t\tnp.mean(1 - p_epoch) = {np.mean(1 - p_epoch)}")
      ACC = ACC + max (0.0, np.mean(1 - p_epoch) - Delta_old)
      ACC = min(1.0,ACC)
      

      varepsilon[test_ind,epoch] =  np.maximum(np.mean(nu_epoch) - ( ( rho_hat * ( 1 - Delta) ) ),0)
      EV_sm[test_ind,epoch] =  mu_1
      KL_sm[test_ind,epoch] = Kullback_Leibler(mu =  mu_1 , sigma = sigma_1 , m = 1.0 , s = sigma_one_sm_train )
      KL_evm[test_ind,epoch] = Kullback_Leibler(mu =  mu_2 , sigma = sigma_2 , m = 1.0 , s = sigma_one_evm_train )
      KL_short[test_ind,epoch] = Bi_Kullback_Leibler(mu_1 = mu_1, mu_2 = mu_2, sigma_1 = sigma_1, 
                                     sigma_2 = sigma_2, rho = 0.0, m_1 = mu_sm_train , m_2 = mu_evm_train, 
                                     s_1 = sigma_sm_train, s_2 = sigma_evm_train, r = 0.0)
      KL_full[test_ind,epoch] = Bi_Kullback_Leibler(mu_1 = mu_1, mu_2 = mu_2, sigma_1 = sigma_1, 
                                     sigma_2 = sigma_2, rho = rho_epoch, m_1 = mu_sm_train , m_2 = mu_evm_train, 
                                     s_1 = sigma_sm_train, s_2 = sigma_evm_train, r = corr_train)
      old_ond[test_ind,epoch]  = ACC

    counter_all = 0
    counter_known = 0
    counter_unknown = 0
    for epoch in range(20,40):
      sm_epoch = np.zeros(batch_size)
      p_epoch = np.zeros(batch_size)
      for i in range(batch_size):
        if is_known_2[counter_all] == 1:
          sm_epoch[i] = sm_known_2[counter_known]
          p_epoch[i] = p_known_2[counter_known]
          counter_known = counter_known + 1
        else:
          sm_epoch[i] = sm_unknown_2[counter_unknown]
          p_epoch[i] = p_unknown_2[counter_unknown]
          counter_unknown = counter_unknown + 1
        counter_all = counter_all + 1
      mu_1 = np.mean(sm_epoch)
      mu_2 = np.mean(p_epoch)
      sigma_1 = np.std(sm_epoch)
      sigma_2 = np.std(p_epoch)
      rho_epoch = stats.pearsonr(sm_epoch, p_epoch)[0]
      
      v_epoch = 1 - Delta - p_epoch
      nu_epoch = np.maximum(v_epoch,np.zeros(batch_size))
      #print(f"{epoch=}\t\tnp.mean(1 - p_epoch) = {np.mean(1 - p_epoch)}")
      ACC = ACC + max (0.0, np.mean(1 - p_epoch) - Delta_old)
      ACC = min(1.0,ACC)

      varepsilon[test_ind,epoch] =  np.maximum(np.mean(nu_epoch) - ( ( rho_hat * ( 1 - Delta) ) ),0)
      EV_sm[test_ind,epoch] =  mu_1
      KL_sm[test_ind,epoch] = Kullback_Leibler(mu =  mu_1 , sigma = sigma_1 , m = 1.0 , s = sigma_one_sm_train )
      KL_evm[test_ind,epoch] = Kullback_Leibler(mu =  mu_2 , sigma = sigma_2 , m = 1.0 , s = sigma_one_evm_train )
      KL_short[test_ind,epoch] = Bi_Kullback_Leibler(mu_1 = mu_1, mu_2 = mu_2, sigma_1 = sigma_1, 
                                     sigma_2 = sigma_2, rho = 0.0, m_1 = mu_sm_train , m_2 = mu_evm_train, 
                                     s_1 = sigma_sm_train, s_2 = sigma_evm_train, r = 0.0)
      KL_full[test_ind,epoch] = Bi_Kullback_Leibler(mu_1 = mu_1, mu_2 = mu_2, sigma_1 = sigma_1, 
                                     sigma_2 = sigma_2, rho = rho_epoch, m_1 = mu_sm_train , m_2 = mu_evm_train, 
                                     s_1 = sigma_sm_train, s_2 = sigma_evm_train, r = corr_train)
      old_ond[test_ind,epoch]  = ACC     




  print("median EV_sm before = ", np.median( EV_sm[:,:20]))
  print("median EV_sm after = ", np.median( EV_sm[:,20:]))
  print("median KL_sm before = ", np.median( KL_sm[:,:20]))
  print("median KL_sm after = ", np.median( KL_sm[:,20:]))
  print("median varepsilon before = ", np.median( varepsilon[:,:20]))
  print("median varepsilon after = ", np.median( varepsilon[:,20:]))
  print("median KL_evm before = ", np.median( KL_evm[:,:20]))
  print("median KL_evm after = ", np.median( KL_evm[:,20:]))
  print("median KL_short before = ", np.median( KL_short[:,:20]))
  print("median KL_short after = ", np.median( KL_short[:,20:]))
  print("median KL_full before = ", np.median( KL_full[:,:20]))
  print("median KL_full after = ", np.median( KL_full[:,20:]))
  print("median old_ond before = ", np.median( old_ond[:,:20]))
  print("median old_ond after = ", np.median( old_ond[:,20:]))


  print("min EV_sm before = ", np.min( EV_sm[:,:20]))
  print("max KL_sm before = ", np.max( KL_sm[:,:20]))
  print("max varepsilon before = ", np.max( varepsilon[:,:20]))
  print("max KL_evm before = ", np.max( KL_evm[:,:20]))
  print("max KL_short before = ", np.max( KL_short[:,:20]))
  print("max KL_full before = ", np.max( KL_full[:,:20]))
  print("max old_ond before = ", np.max( old_ond[:,:20]))
  
  
  interval_EV_sm = np.linspace(0.6, 1.0, num = number_of_points)
  interval_KL_sm = np.linspace(0.0, 1.0, num = number_of_points)
  interval_varepsilon = np.linspace(0.0, 0.25, num = number_of_points)
  interval_KL_evm = np.linspace(10.0, 40.0, num = number_of_points)
  interval_KL_short = np.linspace(5.0, 40.0, num = number_of_points)
  interval_KL_full = np.linspace(5.0, 40.0, num = number_of_points)
  interval_old_ond = np.linspace(0.0, 1.0, num = number_of_points)

  for point in range(number_of_points):
    thresh_EV_sm = interval_EV_sm[point]
    thresh_KL_sm = interval_KL_sm[point]
    thresh_varepsilon = interval_varepsilon[point]
    thresh_KL_evm = interval_KL_evm[point]
    thresh_KL_short = interval_KL_short[point]
    thresh_KL_full = interval_KL_full[point]
    thresh_old_ond = interval_old_ond[point]
    
    prediction_EV_sm = np.copy((EV_sm>thresh_EV_sm).astype(int))
    prediction_KL_sm = np.copy((KL_sm>thresh_KL_sm).astype(int))
    prediction_varepsilon = np.copy((varepsilon>thresh_varepsilon).astype(int))
    prediction_KL_evm = np.copy((KL_evm>thresh_KL_evm).astype(int))
    prediction_KL_short = np.copy((KL_short>thresh_KL_short).astype(int))
    prediction_KL_full = np.copy((KL_full>thresh_KL_full).astype(int))
    prediction_old_ond = np.copy((old_ond>thresh_old_ond).astype(int))
  
  
    for i in range(number_of_tests):
      for j in range(1,40):
        prediction_EV_sm[i,j] = max(prediction_EV_sm[i,j-1] , prediction_EV_sm[i,j])
        prediction_KL_sm[i,j] = max(prediction_KL_sm[i,j-1] , prediction_KL_sm[i,j])
        prediction_varepsilon[i,j] = max(prediction_varepsilon[i,j-1] , prediction_varepsilon[i,j])
        prediction_KL_evm[i,j] = max(prediction_KL_evm[i,j-1] , prediction_KL_evm[i,j])
        prediction_KL_short[i,j] = max(prediction_KL_short[i,j-1] , prediction_KL_short[i,j])
        prediction_KL_full[i,j] = max(prediction_KL_full[i,j-1] , prediction_KL_full[i,j])
        prediction_old_ond[i,j] = max(prediction_old_ond[i,j-1] , prediction_old_ond[i,j])

    correctness_EV_sm= np.zeros((number_of_tests,40))
    correctness_KL_sm = np.zeros((number_of_tests,40))
    correctness_varepsilon = np.zeros((number_of_tests,40))
    correctness_KL_evm = np.zeros((number_of_tests,40))
    correctness_KL_short = np.zeros((number_of_tests,40))
    correctness_KL_full = np.zeros((number_of_tests,40))
    correctness_old_ond = np.zeros((number_of_tests,40))

    for i in range(number_of_tests):
      for j in range(20):
        if prediction_EV_sm[i,j] == 0 :
          correctness_EV_sm[i,j] = 1
        if prediction_KL_sm[i,j] == 0 :
          correctness_KL_sm[i,j] = 1
        if prediction_varepsilon[i,j] == 0 :
          correctness_varepsilon[i,j] = 1
        if prediction_KL_evm[i,j] == 0 :
          correctness_KL_evm[i,j] = 1
        if prediction_KL_short[i,j] == 0 :
          correctness_KL_short[i,j] = 1
        if prediction_KL_full[i,j] == 0 :
          correctness_KL_full[i,j] = 1
        if prediction_old_ond[i,j] == 0 :
          correctness_old_ond[i,j] = 1

      for j in range(20,40):
        if prediction_EV_sm[i,j] == 1 :
          correctness_EV_sm[i,j] = 1    
        if prediction_KL_sm[i,j] == 1 :
          correctness_KL_sm[i,j] = 1    
        if prediction_varepsilon[i,j] == 1 :
          correctness_varepsilon[i,j] = 1 
        if prediction_KL_evm[i,j] == 1 :
          correctness_KL_evm[i,j] = 1    
        if prediction_KL_short[i,j] == 1 :
          correctness_KL_short[i,j] = 1    
        if prediction_KL_full[i,j] == 1 :
          correctness_KL_full[i,j] = 1    
        if prediction_old_ond[i,j] == 1 :
          correctness_old_ond[i,j] = 1    

    prediction_pre_EV_sm  = prediction_EV_sm[:,:20]
    prediction_pre_KL_sm  = prediction_KL_sm[:,:20]
    prediction_pre_varepsilon  = prediction_varepsilon[:,:20]
    prediction_pre_KL_evm  = prediction_KL_evm[:,:20]
    prediction_pre_KL_short  = prediction_KL_short[:,:20]
    prediction_pre_KL_full  = prediction_KL_full[:,:20]
    prediction_pre_old_ond  = prediction_old_ond[:,:20]
    prediction_post_EV_sm = prediction_EV_sm[:,20:] 
    prediction_post_KL_sm = prediction_KL_sm[:,20:] 
    prediction_post_varepsilon = prediction_varepsilon[:,20:] 
    prediction_post_KL_evm = prediction_KL_evm[:,20:] 
    prediction_post_KL_short = prediction_KL_short[:,20:] 
    prediction_post_KL_full = prediction_KL_full[:,20:] 
    prediction_post_old_ond = prediction_old_ond[:,20:] 
    correctness_pre_EV_sm  = correctness_EV_sm[:,:20]
    correctness_pre_KL_sm  = correctness_KL_sm[:,:20]
    correctness_pre_varepsilon  = correctness_varepsilon[:,:20]
    correctness_pre_KL_evm  = correctness_KL_evm[:,:20]
    correctness_pre_KL_short  = correctness_KL_short[:,:20]
    correctness_pre_KL_full  = correctness_KL_full[:,:20]
    correctness_pre_old_ond  = correctness_old_ond[:,:20]
    correctness_post_EV_sm = correctness_EV_sm[:,20:] 
    correctness_post_KL_sm = correctness_KL_sm[:,20:] 
    correctness_post_varepsilon = correctness_varepsilon[:,20:] 
    correctness_post_KL_evm = correctness_KL_evm[:,20:] 
    correctness_post_KL_short = correctness_KL_short[:,20:] 
    correctness_post_KL_full = correctness_KL_full[:,20:] 
    correctness_post_old_ond = correctness_old_ond[:,20:] 

    threshold[index_rho,point,0] = thresh_EV_sm
    threshold[index_rho,point,1] = thresh_KL_sm
    threshold[index_rho,point,2] = thresh_varepsilon
    threshold[index_rho,point,3] = thresh_KL_evm
    threshold[index_rho,point,4] = thresh_KL_short
    threshold[index_rho,point,5] = thresh_KL_full
    threshold[index_rho,point,6] = thresh_old_ond
    
    total_accuracy[index_rho,point,0] = np.sum(correctness_EV_sm) / np.prod(correctness_EV_sm.shape)
    total_accuracy[index_rho,point,1] = np.sum(correctness_KL_sm) / np.prod(correctness_KL_sm.shape)
    total_accuracy[index_rho,point,2] = np.sum(correctness_varepsilon) / np.prod(correctness_varepsilon.shape)
    total_accuracy[index_rho,point,3] = np.sum(correctness_KL_evm) / np.prod(correctness_KL_evm.shape)
    total_accuracy[index_rho,point,4] = np.sum(correctness_KL_short) / np.prod(correctness_KL_short.shape)
    total_accuracy[index_rho,point,5] = np.sum(correctness_KL_full) / np.prod(correctness_KL_full.shape)
    total_accuracy[index_rho,point,6] = np.sum(correctness_old_ond) / np.prod(correctness_old_ond.shape)
    
    pre_accuracy[index_rho,point,0] = np.sum(correctness_pre_EV_sm) / np.prod(correctness_pre_EV_sm.shape)
    pre_accuracy[index_rho,point,1] = np.sum(correctness_pre_KL_sm) / np.prod(correctness_pre_KL_sm.shape)
    pre_accuracy[index_rho,point,2] = np.sum(correctness_pre_varepsilon) / np.prod(correctness_pre_varepsilon.shape)
    pre_accuracy[index_rho,point,3] = np.sum(correctness_pre_KL_evm) / np.prod(correctness_pre_KL_evm.shape)
    pre_accuracy[index_rho,point,4] = np.sum(correctness_pre_KL_short) / np.prod(correctness_pre_KL_short.shape)
    pre_accuracy[index_rho,point,5] = np.sum(correctness_pre_KL_full) / np.prod(correctness_pre_KL_full.shape)
    pre_accuracy[index_rho,point,6] = np.sum(correctness_pre_old_ond) / np.prod(correctness_pre_old_ond.shape)
    
    
    post_accuracy[index_rho,point,0] = np.sum(correctness_post_EV_sm) / np.prod(correctness_post_EV_sm.shape)
    post_accuracy[index_rho,point,1] = np.sum(correctness_post_KL_sm) / np.prod(correctness_post_KL_sm.shape)
    post_accuracy[index_rho,point,2] = np.sum(correctness_post_varepsilon) / np.prod(correctness_post_varepsilon.shape)
    post_accuracy[index_rho,point,3] = np.sum(correctness_post_KL_evm) / np.prod(correctness_post_KL_evm.shape)
    post_accuracy[index_rho,point,4] = np.sum(correctness_post_KL_short) / np.prod(correctness_post_KL_short.shape)
    post_accuracy[index_rho,point,5] = np.sum(correctness_post_KL_full) / np.prod(correctness_post_KL_full.shape)
    post_accuracy[index_rho,point,6] = np.sum(correctness_post_old_ond) / np.prod(correctness_post_old_ond.shape)


    failure[index_rho,point,0] =  np.sum((1 - prediction_post_EV_sm[:,-1]) / number_of_tests)
    failure[index_rho,point,1] =  np.sum((1 - prediction_post_KL_sm[:,-1]) / number_of_tests)
    failure[index_rho,point,2] =  np.sum((1 - prediction_post_varepsilon[:,-1]) / number_of_tests)
    failure[index_rho,point,3] =  np.sum((1 - prediction_post_KL_evm[:,-1]) / number_of_tests)
    failure[index_rho,point,4] =  np.sum((1 - prediction_post_KL_short[:,-1]) / number_of_tests)
    failure[index_rho,point,5] =  np.sum((1 - prediction_post_KL_full[:,-1]) / number_of_tests)
    failure[index_rho,point,6] =  np.sum((1 - prediction_post_old_ond[:,-1]) / number_of_tests)


    early[index_rho,point,0] =  np.sum(prediction_pre_EV_sm[:,-1])/ number_of_tests
    early[index_rho,point,1] =  np.sum(prediction_pre_KL_sm[:,-1])/ number_of_tests
    early[index_rho,point,2] =  np.sum(prediction_pre_varepsilon[:,-1])/ number_of_tests
    early[index_rho,point,3] =  np.sum(prediction_pre_KL_evm[:,-1])/ number_of_tests
    early[index_rho,point,4] =  np.sum(prediction_pre_KL_short[:,-1])/ number_of_tests
    early[index_rho,point,5] =  np.sum(prediction_pre_KL_full[:,-1])/ number_of_tests
    early[index_rho,point,6] =  np.sum(prediction_pre_old_ond[:,-1])/ number_of_tests


    on_time[index_rho,point,0] = np.sum( ( (1 - prediction_pre_EV_sm[:,-1]) * prediction_post_EV_sm[:,0]) ) / number_of_tests
    on_time[index_rho,point,1] = np.sum( ( (1 - prediction_pre_KL_sm[:,-1]) * prediction_post_KL_sm[:,0]) ) / number_of_tests
    on_time[index_rho,point,2] = np.sum( ( (1 - prediction_pre_varepsilon[:,-1]) * prediction_post_varepsilon[:,0]) ) / number_of_tests
    on_time[index_rho,point,3] = np.sum( ( (1 - prediction_pre_KL_evm[:,-1]) * prediction_post_KL_evm[:,0]) ) / number_of_tests
    on_time[index_rho,point,4] = np.sum( ( (1 - prediction_pre_KL_short[:,-1]) * prediction_post_KL_short[:,0]) ) / number_of_tests
    on_time[index_rho,point,5] = np.sum( ( (1 - prediction_pre_KL_full[:,-1]) * prediction_post_KL_full[:,0]) ) / number_of_tests
    on_time[index_rho,point,6] = np.sum( ( (1 - prediction_pre_old_ond[:,-1]) * prediction_post_old_ond[:,0]) ) / number_of_tests


    late[index_rho,point,0] = np.sum( ( 1 - prediction_post_EV_sm[:,0] ) * prediction_post_EV_sm[:,-1]) / number_of_tests
    late[index_rho,point,1] = np.sum( ( 1 - prediction_post_KL_sm[:,0] ) * prediction_post_KL_sm[:,-1]) / number_of_tests
    late[index_rho,point,2] = np.sum( ( 1 - prediction_post_varepsilon[:,0] ) * prediction_post_varepsilon[:,-1]) / number_of_tests
    late[index_rho,point,3] = np.sum( ( 1 - prediction_post_KL_evm[:,0] ) * prediction_post_KL_evm[:,-1]) / number_of_tests
    late[index_rho,point,4] = np.sum( ( 1 - prediction_post_KL_short[:,0] ) * prediction_post_KL_short[:,-1]) / number_of_tests
    late[index_rho,point,5] = np.sum( ( 1 - prediction_post_KL_full[:,0] ) * prediction_post_KL_full[:,-1]) / number_of_tests
    late[index_rho,point,6] = np.sum( ( 1 - prediction_post_old_ond[:,0] ) * prediction_post_old_ond[:,-1]) / number_of_tests


    early_count_EV_sm = np.count_nonzero(prediction_pre_EV_sm, axis = 1) * prediction_post_EV_sm[:,-1]
    early_count_KL_sm = np.count_nonzero(prediction_pre_KL_sm, axis = 1) * prediction_post_KL_sm[:,-1]
    early_count_varepsilon = np.count_nonzero(prediction_pre_varepsilon, axis = 1) * prediction_post_varepsilon[:,-1]
    early_count_KL_evm = np.count_nonzero(prediction_pre_KL_evm, axis = 1) * prediction_post_KL_evm[:,-1]
    early_count_KL_short = np.count_nonzero(prediction_pre_KL_short, axis = 1) * prediction_post_KL_short[:,-1]
    early_count_KL_full = np.count_nonzero(prediction_pre_KL_full, axis = 1) * prediction_post_KL_full[:,-1]
    early_count_old_ond = np.count_nonzero(prediction_pre_old_ond, axis = 1) * prediction_post_old_ond[:,-1]
    
    
    late_count_EV_sm = (20 - np.count_nonzero(prediction_post_EV_sm, axis = 1)) * prediction_post_EV_sm[:,-1]    
    late_count_KL_sm = (20 - np.count_nonzero(prediction_post_KL_sm, axis = 1)) * prediction_post_KL_sm[:,-1]    
    late_count_varepsilon = (20 - np.count_nonzero(prediction_post_varepsilon, axis = 1)) * prediction_post_varepsilon[:,-1]    
    late_count_KL_evm = (20 - np.count_nonzero(prediction_post_KL_evm, axis = 1)) * prediction_post_KL_evm[:,-1]    
    late_count_KL_short = (20 - np.count_nonzero(prediction_post_KL_short, axis = 1)) * prediction_post_KL_short[:,-1]    
    late_count_KL_full = (20 - np.count_nonzero(prediction_post_KL_full, axis = 1)) * prediction_post_KL_full[:,-1]    
    late_count_old_ond = (20 - np.count_nonzero(prediction_post_old_ond, axis = 1)) * prediction_post_old_ond[:,-1]    


    absolute_count_EV_sm = np.sum( early_count_EV_sm + late_count_EV_sm )
    absolute_count_KL_sm = np.sum( early_count_KL_sm + late_count_KL_sm )
    absolute_count_varepsilon = np.sum( early_count_varepsilon + late_count_varepsilon )
    absolute_count_KL_evm = np.sum( early_count_KL_evm + late_count_KL_evm )
    absolute_count_KL_short = np.sum( early_count_KL_short + late_count_KL_short )
    absolute_count_KL_full = np.sum( early_count_KL_full + late_count_KL_full )
    absolute_count_old_ond = np.sum( early_count_old_ond + late_count_old_ond )
    
    
    not_fail_count_EV_sm = np.count_nonzero(prediction_EV_sm[:,-1])
    not_fail_count_KL_sm = np.count_nonzero(prediction_KL_sm[:,-1])
    not_fail_count_varepsilon= np.count_nonzero(prediction_varepsilon[:,-1])
    not_fail_count_KL_evm = np.count_nonzero(prediction_KL_evm[:,-1])
    not_fail_count_KL_short = np.count_nonzero(prediction_KL_short[:,-1])
    not_fail_count_KL_full = np.count_nonzero(prediction_KL_full[:,-1])
    not_fail_count_old_ond = np.count_nonzero(prediction_old_ond[:,-1])
    
    
    
    if not_fail_count_EV_sm > 0:
      absolute_error[index_rho,point,0] = absolute_count_EV_sm / not_fail_count_EV_sm
    else:
      absolute_error[index_rho,point,0] = 0
    if not_fail_count_KL_sm > 0:
      absolute_error[index_rho,point,1] = absolute_count_KL_sm / not_fail_count_KL_sm
    else:
      absolute_error[index_rho,point,1] = 0
    if not_fail_count_varepsilon > 0:
      absolute_error[index_rho,point,2] = absolute_count_varepsilon / not_fail_count_varepsilon
    else:
      absolute_error[index_rho,point,2] = 0
    if not_fail_count_KL_evm > 0:
      absolute_error[index_rho,point,3] = absolute_count_KL_evm / not_fail_count_KL_evm
    else:
      absolute_error[index_rho,point,3] = 0
    if not_fail_count_KL_short > 0:
      absolute_error[index_rho,point,4] = absolute_count_KL_short/ not_fail_count_KL_short
    else:
      absolute_error[index_rho,point,4] = 0
    if not_fail_count_KL_full > 0:
      absolute_error[index_rho,point,5] = absolute_count_KL_full / not_fail_count_KL_full
    else:
      absolute_error[index_rho,point,5] = 0
    if not_fail_count_old_ond  > 0:
      absolute_error[index_rho,point,6] = absolute_count_old_ond  / not_fail_count_old_ond 
    else:
      absolute_error[index_rho,point,6] = 0


np.savez('ond_array_plot_4.npz', threshold=threshold, total_accuracy=total_accuracy, pre_accuracy=pre_accuracy,
         post_accuracy=post_accuracy, failure=failure, early=early, on_time=on_time, late=late,
         absolute_error= absolute_error)


