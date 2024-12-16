from torchvision.transforms.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch 

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

def entropy(logits):
    #if ensemble do not use softmax, else use softmax 
    probs = F.softmax(logits, dim=1)
    #probs = logits
    log_probs = torch.log(probs+1e-10)
    #print(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


def pavpu(logits, labels, w, stride, certain_t, acc_t):
    # Compute uncertainty mask using entropy
    uncertainty_mask = entropy(logits).detach().cpu()
    labels = labels.detach().cpu().squeeze()
    pred_mask = torch.argmax(logits.squeeze(), dim=0).detach().cpu()
    b_mask = pred_mask.eq(labels).float()
    
    # Prepare unfolded patches
    patches_uncertainty = uncertainty_mask.unfold(1, w, stride).unfold(2, w, stride)
    patches_accuracy = b_mask.unfold(0, w, stride).unfold(1, w, stride)

    # Reshape patches to a 2D array of patches
    patches_uncertainty = patches_uncertainty.contiguous().view(-1, w*w)
    patches_accuracy = patches_accuracy.contiguous().view(-1, w*w)
    
    # Calculate average uncertainty and accuracy for each patch
    avg_uncertainty = patches_uncertainty.mean(dim=1)
    avg_accuracy = patches_accuracy.mean(dim=1)

    # Classify each patch
    k_1 = avg_accuracy >= acc_t  # 'a' if true, 'i' if false
    k_2 = avg_uncertainty <= certain_t  # 'c' if true, 'u' if false

    # Create a table for counts
    tab = {
        'a': {'c': torch.sum(k_1 & k_2).item(), 'u': torch.sum(k_1 & ~k_2).item()},
        'i': {'c': torch.sum(~k_1 & k_2).item(), 'u': torch.sum(~k_1 & ~k_2).item()}
    }

    # Calculate PAvPU
    pavpu_value = (tab['a']['c'] + tab['i']['u']) / (tab['a']['c'] + tab['a']['u'] + tab['i']['c'] + tab['i']['u'])
    return pavpu_value 


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
