import torch
import numpy as np
import torch.nn.functional as F

class OcclusionExplainer:
    def __init__(self, model):
        self.model = model

    def generate(self, image_tensor, class_idx, patch=32):
        _, _, H, W = image_tensor.shape

        with torch.no_grad():
            base_score = torch.softmax(self.model(image_tensor), dim=1)[0, class_idx].item()

        heatmap = torch.zeros((H, W)).to(image_tensor.device)

        for y in range(0, H, patch):
            for x in range(0, W, patch):
                occluded = image_tensor.clone()
                occluded[:, :, y:y+patch, x:x+patch] = 0

                with torch.no_grad():
                    score = torch.softmax(self.model(occluded), dim=1)[0, class_idx].item()

                drop = base_score - score
                heatmap[y:y+patch, x:x+patch] = drop

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap
