import numpy as np
import matplotlib.pyplot as plt

data  = np.load('ond_window_plot_1.npz')

threshold = data['threshold']
total_accuracy = data['total_accuracy']
failure = data['failure'] 
early = data['early']
on_time = data['on_time']
late = data['late']
absolute_error = data['MAE']
del data


detected = 1 - failure
x = np.arange(2,26)

on_and_late = on_time + late



ind = [0] * 4
for k in range(4):
  try:
    #ind[k] = np.argmax(total_accuracy[0,:,k])
    #ind[k] = np.argmax(on_and_late[0,:,k])
    #ind[k] = np.nonzero(early[0,:,k] < 0.01)[0][0]
    ind[k] = np.nonzero(early[0,:,k] < 0.05)[0][0]
  except:
    print(f"Error in k  = {k}")
    k = -1


score = detected * on_time
'''
absolute_error[failure > 0.99] = np.nan
early[failure > 0.99] = np.nan
on_time[failure > 0.99] = np.nan
late[failure > 0.99] = np.nan
total_accuracy[failure > 0.99] = np.nan
detected[failure > 0.99] = np.nan
score[failure > 0.99] = np.nan
on_and_late[failure > 0.99] = np.nan
'''


  
peak_total_accuracy = np.zeros((len(x),failure.shape[2]))
peak_early = np.zeros((len(x),failure.shape[2]))
peak_on_time = np.zeros((len(x),failure.shape[2]))
peak_late = np.zeros((len(x),failure.shape[2]))
peak_absolute_error = np.zeros((len(x),failure.shape[2]))
peak_detected = np.zeros((len(x),failure.shape[2]))
peak_score = np.zeros((len(x),failure.shape[2]))
peak_on_and_late = np.zeros((len(x),failure.shape[2]))

for k,p in enumerate(x):
  for j in range(4):
    peak_total_accuracy[k,j] = total_accuracy[k,ind[j],j]
    peak_early[k,j] = early[k,ind[j],j]
    peak_on_time[k,j] = on_time[k,ind[j],j]
    peak_late[k,j] = late[k,ind[j],j]
    peak_absolute_error[k,j] = absolute_error[k,ind[j],j]
    peak_detected[k,j] = detected[k,ind[j],j]
    peak_score[k,j] = score[k,ind[j],j]
    peak_on_and_late[k,j] = on_and_late[k,ind[j],j]
  


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_total_accuracy[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_total_accuracy[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_total_accuracy[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_total_accuracy[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([50.0,100.0])
# plt.xlabel('Percentage of unknown')
plt.ylabel('Accuracy (%)')
plt.title(f'Total Accuracy')
plt.legend(loc='upper left' , ncol=2)
plt.show()




fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_detected[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_detected[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_detected[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_detected[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([00.0,100.0])
# plt.xlabel('Percentage of unknown')
plt.ylabel('Percentage')
plt.title('Total Detection Percentage')
# plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.legend(loc='lower right' , ncol=2)
plt.show()


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_on_and_late[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_on_and_late[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_on_and_late[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_on_and_late[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([0.0,100.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Percenage')
plt.title('True Detection Percentage')
plt.legend(loc='lower right' , ncol=2)
plt.show()



fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_early[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_early[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_early[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_early[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([0.0,100.0])
# plt.xlabel('Percentage of unknown')
plt.ylabel('Percentage')
plt.title('Percentage of early')
# plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.legend(loc='upper center' , ncol=2)
plt.show()




fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_on_time[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_on_time[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_on_time[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_on_time[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([0.0,100.0])
# plt.xlabel('Percentage of unknown')
plt.ylabel('Percentage')
plt.title('Percentage of on-time')
# plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.legend(loc='upper center' , ncol=2)
plt.show()



fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_late[:,0] * 100, 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_late[:,1] * 100, 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_late[:,2] * 100, '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_late[:,3] * 100, 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([0.0,100.0])
# plt.xlabel('Percentage of unknown')
plt.ylabel('Percentage')
plt.title('Percentage of late')
# plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.legend(loc='lower right' , ncol=2)
plt.show()






fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_absolute_error[:,0], 'o', color='#ff0000', linewidth=3, label = 'KL SoftMax')
ax.plot(x, peak_absolute_error[:,1], 's', color='#008000', linewidth=3, label = 'KL EVM')
ax.plot(x, peak_absolute_error[:,2], '*', color='#ff8080', linewidth=3, label = 'ACC SoftMax')
ax.plot(x, peak_absolute_error[:,3], 'D', color='#00ff00', linewidth=3, label = 'ACC EVM')
ax.set_xlim([1,26])
ax.set_ylim([0.0,1000.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Error')
plt.title('Mean Absolute Error')
# plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.legend(loc='lower left' , ncol=2)
plt.show()
