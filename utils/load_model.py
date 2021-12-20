import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
USE_FUNCTION={}
def load_model(path,model):
    model_dict=torch.load(path)["snn_state_dict"]
    model_state_dict=model.state_dict()
    pretrained_dict = {k: v for k, v in model_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    for k,v in pretrained_dict.items():
        if "norm" in k and "num_batches_tracked" not  in k:
            pretrained_dict[k].requires_grad=True
        elif "turn_layer" in k and "conv" in k:
            pretrained_dict[k].requires_grad=True
        else:
            continue
            #pretrained_dict[k].requires_grad=True
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model
def save_log(**kwargs):
    result=[]
    name=[]
    for key,value in kwargs.items():
        result.append(value)
        name.append(key)
    result=np.array(result).T
    frame=pd.DataFrame(data=result,columns=name)
    time=str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    frame.to_csv("./output/"+time+".csv",index=False,header=True)
