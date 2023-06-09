import torch
import torchvision
from utils.data_loader import get_loader
from utils.utils import show_feature_map
from net.vggface import vggface
import cv2
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def FIAloss(grad, feature):
    # zeros = torch.zeros(feature.shape[1:], dtype=torch.float32).to(device)
    # loss1 = torch.sum(torch.maximum(grad * feature,zeros))
    # loss2 = torch.sum(torch.minimum(grad * feature,zeros))
    # Loss = torch.abs(loss1 - loss2)
    Loss = torch.sum(torch.abs(grad * feature))
    #Loss = torch.sum(grad * feature)
    return Loss
