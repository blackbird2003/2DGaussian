from scene.gaussian_models import Model
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
import torch
from utils.general_utils import compute_gaussian_value

def get_distance(points : torch.Tensor, pixel_coords_batch : torch.Tensor,  batch_size=32):
    B = batch_size
    N = points.shape[0]

    # 扩展像素坐标 [B, 1, 2] -> [B, N, 2]
    pixels_expanded = pixel_coords_batch.unsqueeze(1).expand(B, N, 2).cuda()

    # 扩展点坐标 [1, N, 2] -> [B, N, 2]
    points_expanded = points.unsqueeze(0).expand(B, N, 2).cuda()

    # 计算距离 [B, N, 2]
    diff = pixels_expanded - points_expanded
    # distances = torch.sqrt(torch.sum(diff ** 2, dim=2))

    return diff

def compute_color(points : Model, pixel_coords_batch : torch.Tensor, batch_dist, batch_size=32):
    """
    使用批处理计算所有像素到所有点的距离
    Args:
        points: N 高斯点
        pixel_coords_batch: [B, 2] 批处理像素坐标 [0, 1]
        batch_dist: [B, N, 2] 批处理像素点与每个高斯中心的相对位置
        batch_size: 批处理大小

    Returns:
        pix_color_batch: [B, 3] 批处理颜色
    """
    pix_color_batch = torch.zeros((batch_size, 3)).cuda() # [B, 3]

    return pix_color_batch


def render(pt : Model, opt : OptimizationParams, background, resoulation, batch_size=32):
    torch.cuda.empty_cache()
    H, W = resoulation
    render_img = torch.zeros((resoulation[1], resoulation[0], 3))  # W, H, 3

    points = pt.get_xyz.clone() / torch.tensor([resoulation[0], resoulation[1]]).cuda()  # N, 2

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij'
    )

    # 组合成坐标tensor [H, W, 2]  normalized to [0, 1]
    coords = torch.stack([x_coords / float(H - 1), y_coords / float(W - 1)], dim=-1).float()
    pixels_flat = coords.reshape(-1, 2)  # (H*W), 2
    total_pixels = pixels_flat.shape[0]



    return {"render" : render_img}
