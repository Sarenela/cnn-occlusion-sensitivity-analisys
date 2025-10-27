import torch


def apply_occlusion(image: torch.Tensor, occlusion_percent: float = 20, top: int = None,
                    left: int = None) -> torch.Tensor:
    c, h, w = image.shape
    occ_h = int(h * occlusion_percent / 100)
    occ_w = int(w * occlusion_percent / 100)

    if top is None:
        top = (h - occ_h) // 2
    if left is None:
        left = (w - occ_w) // 2

    occluded = image.clone()
    occluded[:, top:top + occ_h, left:left + occ_w] = 0  #black square

    return occluded
