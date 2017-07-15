import tflearn
import glob
import os
import sys
from models import *
import global_vars_go as gvg
import re

def load_model(model_name):
    try:
        model = eval(model_name)
        if hasattr(model, "get_network"):
            return tflearn.DNN(model.get_network(), tensorboard_verbose=2, checkpoint_path="checkpoints/{}.tflearn".format(model_name))
        else:
            print("ERROR! model, {}, is not defined or does not contain a \"get_network()\" function".format(model_name))
    except NameError:
        print("ERROR! {} is not a valid module".format(model_name))

def load_model_from_file(model_name):
    f = None
    for filename in glob.glob(os.path.join(gvg.checkpoint_path, "*.index")):
        try: #Substring for Windows
            base_filename = filename[filename.index('\\')+1:filename.index('\\')+len(model_name)+1]
        except ValueError: #Substring for Linux
            base_filename = filename[filename.index('/')+1:filename.index('/')+len(model_name)+1]
        if base_filename == model_name and get_int(filename) > get_int(f):
            f = filename[:-6]
    f = f.replace('\\', '/')
    if f == None:
        print("ERROR! There were no saved {} models".format(model_name))
    else:
        model = load_model(model_name)
        model.load(f)
        return model

def get_int(line):
    if line is None:
        return 0

    endl = ""
    for c in line:
        if c.isnumeric():
            endl += c
    return int(endl)
