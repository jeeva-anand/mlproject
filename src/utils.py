
import os
import pickle as pkl

def save_object(path, obj):
  dir = os.path.dirname(path)  
  os.makedirs(dir,exist_ok = True)
  pkl.dump(obj, open(path, "wb"))

