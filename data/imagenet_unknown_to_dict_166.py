import numpy as np
import os
import json

n = 1001
folder_to_id_dict = dict()
with open("new_166.txt", 'r') as f:
  for line in f:
    if line != '':
      print(line)
      folder = line[:-1]
      folder_to_id_dict[folder] = str(n)
      n = n + 1


with open('folder_to_id_dict_unknown_166.json', 'w') as f:
    json.dump(folder_to_id_dict, f)
