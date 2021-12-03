import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
USE_FUNCTION={}
def load_model(path,model):
    model_dict=torch.load(path)["snn_state_dict"]
    model_state_dict=model.state_dict()
    pretrained_dict = {k: v for k, v in model_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    for k,v in pretrained_dict.items():
        if "norm" in k and "num_batches_tracked" not  in k:
            pretrained_dict[k].requires_grad=True
        else:
            pretrained_dict[k].requires_grad=False
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model