from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def tensor_imshow(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0).numpy())
    plt.show()

def tensor_to_image(image_tensor):
    # Convert from CHW to HWC for visualization
    return image_tensor.permute(1, 2, 0).numpy()

def image_to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1)

def _predict_proba(image_batch, model):
    model.eval()
    with torch.torch.no_grad():
        logits = model(image_batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach()  # already in cpu
    
def _single_predict_proba(image_tensor, model):
    # add an extra batch dimension and return the first element
    return _predict_proba(image_tensor.unsqueeze(dim=0), model)[0]


def VisualizeImageGrayscale(image_3d, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor. Useful in visualizing shap explanations.

    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)