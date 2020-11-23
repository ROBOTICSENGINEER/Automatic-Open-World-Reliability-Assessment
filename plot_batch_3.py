import numpy as np
import matplotlib.pyplot as plt

data  = np.load('ond_array_plot_4.npz')

threshold = data['threshold']
total_accuracy = data['total_accuracy']
failure = data['failure'] 
early = data['early']
# on_time = data['on_time']
late = data['late']
absolute_error = data['absolute_error']
del data

index = 5
index = 5

on_time = 1 - (failure+early+late)


detected = 1 - failure
x = threshold[0,:,5]
on_and_late = on_time + late

absolute_error[failure > 0.99] = np.nan
early[failure > 0.99] = np.nan
on_time[failure > 0.99] = np.nan
late[failure > 0.99] = np.nan
total_accuracy[failure > 0.99] = np.nan
detected[failure > 0.99] = np.nan
threshold[failure > 0.99] = np.nan
on_and_late[failure > 0.99] = np.nan

'''
colors= {
   2:'#000000',
   3:'#808080',
   4:'#9999ff',
   5:'#99ff99',
   6:'#ff9999',
   7:'#800080',
   8:'#c000c0',
   9:'#ff00ff',
  10:'#808000',
  11:'#c0c000',
  12:'#ffff00',
  13:'#008080',
  14:'#00c0c0',
  15:'#00ffff',
  16:'#000000',
  17:'#000080',
  18:'#0000c0',
  19:'#0000ff',
  20:'#800000',
  21:'#c00000',
  22:'#ff0000',
  23:'#008000',
  24:'#00c000',
  25:'#00ff00'
  }

colors= {
   2:'#FF00A2',
   3:'#FF00DD',
   4:'#E600FF',
   5:'#AA00FF',
   6:'#6F00FF',
   7:'#3300FF',
   8:'#0008FF',
   9:'#0044FF',
  10:'#0080FF',
  11:'#00BBFF',
  12:'#00F7FF',
  13:'#00FFCC',
  14:'#00FF91',
  15:'#00FF55',
  16:'#00FF1A',
  17:'#22FF00',
  18:'#5EFF00',
  19:'#99FF00',
  20:'#D4FF00',
  21:'#FFEE00',
  22:'#FFB300',
  23:'#FF7700',
  24:'#FF3C00',
  25:'#FF0000'
  }
'''
colors= {
   2:'#FB00FF',
   3:'#C300FF',
   4:'#8C00FF',
   5:'#5500FF',
   6:'#1E00FF',
   7:'#001AFF',
   8:'#0051FF',
   9:'#0088FF',
  10:'#00BFFF',
  11:'#00F7FF',
  12:'#00FFD0',
  13:'#00FF99',
  14:'#00FF62',
  15:'#00FF2A',
  16:'#0DFF00',
  17:'#44FF00',
  18:'#7BFF00',
  19:'#B3FF00',
  20:'#EAFF00',
  21:'#FFDD00',
  22:'#FFA600',
  23:'#FF6F00',
  24:'#FF3700',
  25:'#FF0000'
  }



fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], early[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,100.0])
plt.xlabel('Threshold')
plt.ylabel('Percentage')
# plt.title(f'False Detection Percentage')
plt.legend( ncol=2)
plt.show() 


  
fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], total_accuracy[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([50,100.0])
plt.xlabel('Threshold')
plt.ylabel('Accuracy (%)')
# plt.title(f'Total Accuracy')
plt.show() 


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], detected[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,100.0])
plt.xlabel('Threshold')
plt.ylabel('Percentage')
# plt.title(f'Total Detection Percentage')
plt.show() 


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], on_and_late[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,100.0])
plt.xlabel('Threshold')
plt.ylabel('Percentage')
plt.title(f'True detection percentage')
plt.show() 




fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], on_time[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,100.0])
plt.xlabel('Threshold')
plt.ylabel('Percentage')
# plt.title(f'On-time Percentage')
plt.show() 


fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], late[k,:,index] * 100, color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,100.0])
plt.xlabel('Threshold')
plt.ylabel('Percentage')
# plt.title(f'Late Percentage')
plt.show() 



fig, ax = plt.subplots(figsize=(6,4) , dpi =300)
for k in range(24):
  ax.plot(threshold[k,:,index], absolute_error[k,:,index], color=colors[k+2], linewidth=3, label = f'{k+2}% unknown')
ax.set_xlim([14,36])
# ax.set_ylim([0.62,0.8])
ax.set_ylim([0.0,20.0])
plt.xlabel('Threshold')
plt.ylabel('Error')
# plt.title(f'Mean Absolute Error')
plt.show() 


