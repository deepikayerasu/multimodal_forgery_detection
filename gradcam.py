import cv2
import torch
import numpy as np
from models.video_model import VideoModel

def generate_heatmap(bgr_frame):
    """
    Heuristic saliency over our tiny video CNN.
    Returns a float heatmap in 0..1 with same HxW as input frame.
    """
    model = VideoModel().eval()

    img = cv2.resize(bgr_frame, (299, 299))[:, :, ::-1]  # BGR->RGB
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,299,299]
    x.requires_grad_(True)

    # pass through features and linear head
    feat = model.features(x)                 # [1,64,?,?] -> pooled in model forward
    feat_avg = torch.mean(feat, dim=[2, 3])  # [1,64]
    out = model.fc(feat_avg)                 # [1,256]

    # use L2-norm of output as pseudo-target
    score = out.norm(p=2)
    score.backward()

    grad = x.grad.abs().mean(dim=1)[0].detach().numpy()   # [299,299]
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    grad = cv2.resize(grad, (bgr_frame.shape[1], bgr_frame.shape[0]))
    return grad
