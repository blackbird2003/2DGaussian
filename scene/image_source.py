import torch
from torch import nn
import numpy as np

class Image_source(nn.Module):
    def __init__(self, image : torch.Tensor, id, time):
        super(Image_source, self).__init__()
        self.images = image
        self.id = id
        self.time = time

