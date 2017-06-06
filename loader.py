import tflearn
import sys
from models import *

def load_model(model_name):
    try:
        model = eval(model_name)
        if hasattr(model, "get_network"):
            return tflearn.DNN(model.get_network(), tensorboard_verbose=0, checkpoint_path="checkpoints/{}.ckpt".format(model_name))
        else:
            print("ERROR! model, {}, is not defined or does not contain a \"get_network()\" function".format(model_name))
    except NameError:
        print("ERROR! {} is not a valid module".format(model_name))
