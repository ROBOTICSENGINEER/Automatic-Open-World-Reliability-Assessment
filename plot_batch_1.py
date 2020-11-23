import numpy as np
import matplotlib.pyplot as plt

data  = np.load('ond_array_plot_4.npz')

threshold = data['threshold']
total_accuracy = data['total_accuracy']
failure = data['failure'] 
early = data['early']
on_time = data['on_time']
late = data['late']
absolute_error = data['absolute_error']
del data


detected = 1 - failure
x = np.arange(2,26)


ind = [0] * 7
for k in range(7):
  ind[k] = np.argmax(total_accuracy[0,:,k])


absolute_error[failure > 0.99] = np.nan
early[failure > 0.99] = np.nan
on_time[failure > 0.99] = np.nan
late[failure > 0.99] = np.nan
total_accuracy[failure > 0.99] = np.nan
detected[failure > 0.99] = np.nan

  
peak_total_accuracy = np.zeros((len(x),failure.shape[2]))
peak_early = np.zeros((len(x),failure.shape[2]))
peak_on_time = np.zeros((len(x),failure.shape[2]))
peak_late = np.zeros((len(x),failure.shape[2]))
peak_absolute_error = np.zeros((len(x),failure.shape[2]))
peak_detected = np.zeros((len(x),failure.shape[2]))

for k,p in enumerate(x):
  for j in range(7):
    peak_total_accuracy[k,j] = total_accuracy[k,ind[j],j]
    peak_early[k,j] = early[k,ind[j],j]
    peak_on_time[k,j] = on_time[k,ind[j],j]
    peak_late[k,j] = late[k,ind[j],j]
    peak_absolute_error[k,j] = absolute_error[k,ind[j],j]
    peak_detected[k,j] = detected[k,ind[j],j]
  


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_total_accuracy[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_total_accuracy[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_total_accuracy[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_total_accuracy[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_total_accuracy[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_total_accuracy[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_total_accuracy[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Accuracy')
plt.title(f'Total Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_early[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_early[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_early[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_early[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_early[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_early[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_early[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Ratio')
plt.title('Ratio early')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()



fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_on_time[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_on_time[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_on_time[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_on_time[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_on_time[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_on_time[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_on_time[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Ratio')
plt.title('Ratio on-time')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_late[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_late[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_late[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_late[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_late[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_late[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_late[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Ratio')
plt.title('Ratio late')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()




fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_absolute_error[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_absolute_error[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_absolute_error[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_absolute_error[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_absolute_error[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_absolute_error[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_absolute_error[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Error')
plt.title('Mean Absolute Error')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
ax.plot(x, peak_detected[:,0], color='#800000', linewidth=4, label = 'mean SoftMax')
ax.plot(x, peak_detected[:,1], color='#ff0000', linewidth=4, label = 'KL SoftMax')
# ax.plot(x, peak_detected[:,6], color='#ff00ff', linewidth=4, label = 'OLD OND')
ax.plot(x, peak_detected[:,2], color='#0000ff', linewidth=4, label = 'OND EVM')
ax.plot(x, peak_detected[:,3], color='#00ffff', linewidth=4, label = 'KL EVM')
# ax.plot(x, peak_detected[:,4], color='#00ff00', linewidth=4, label = 'Bi KL independent')
ax.plot(x, peak_detected[:,5], color='#008000', linewidth=4, label = 'Bi KL full')
ax.set_xlim([2,25])
# ax.set_ylim([0.0,1.0])
plt.xlabel('Percentage of unknown')
plt.ylabel('Ratio')
plt.title('Ratio Detected')
plt.legend(bbox_to_anchor=(1.05, 1) , ncol=1)
plt.show()
