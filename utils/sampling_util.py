import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import torch

def get_sample_init_xy(n_points=300):
    """
    使用Sobol序列生成初始采样点
    :param resolution: 图像分辨率 (width, height)
    :param n_points: 采样点数量
    :return: 采样点坐标列表 [(x1, y1), (x2, y2), ...]
    """
    sobol = qmc.Sobol(d=2, scramble=True)
    samples = sobol.random(n=n_points)

    samples = torch.from_numpy(samples).float().cuda()

    return samples

