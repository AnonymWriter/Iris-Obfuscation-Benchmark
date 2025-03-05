import os
import torch
import random
import shutil
import numpy as np
import polarTransform
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score} #, 'mcc' : matthews_corrcoef}

def seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def prepare_dir(dir: str) -> None:
    """
    Prepare the directory for savings.

    Arguments:
        dir (str): directory path.
    """

    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def crop_image(image: torch.Tensor, return_idx: bool = False) -> torch.Tensor | tuple[int, int, int, int]:
    """
    Trim the black border of an image.
    
    Arguments:
        image (torch.Tensor): image tensor.
        return_idx (bool): whether to return trimmed image, or the trimming bounding box.

    Returns:
        x_min, y_min, x_max, y_max (tuple[int, int, int, int]): trimming bounding box.
        cropped (torch.Tensor): trimmed image tensor.
    """

    nonzero = image.nonzero()

    if len(image.shape) == 2: # image of shape (h, w)
        x_min, y_min = nonzero.min(dim = 0)[0]
        x_max, y_max = nonzero.max(dim = 0)[0]
    elif len(image.shape) == 3 and image.shape[0] == 1: # image of shape (1, h, w)
        _, x_min, y_min = nonzero.min(dim = 0)[0]
        _, x_max, y_max = nonzero.max(dim = 0)[0]
    else:
        raise Exception('image shape wrong:', image.shape)

    if return_idx:
        return x_min, y_min, x_max, y_max
    else:
        cropped = image[:, x_min: x_max + 1, y_min: y_max + 1]
        return cropped
    
def mask_images(images: list[torch.Tensor], masks: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Apply mask to images.

    Arguments:
        images (list[torch.Tensor]): list of images.
        masks (list[torch.Tensor]): list of masks.

    Returns:
        masked_images (list[torch.Tensor]): list of masked images.
    """

    masked_images = []
    for image, mask in zip(images, masks):
        masked_images.append(image * mask)
    return masked_images

def replace_image_region(images: list[torch.Tensor], new_regions: list[torch.Tensor], region_masks: list[torch.Tensor], bounding_boxes: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Replace regions in images with new regions.
    
    Arguments:
        images (list[torch.Tensor]): list of images.
        new_regions (list[torch.Tensor]): list of new regions.
        region_masks (list[torch.Tensor]): list of region masks.
        bounding_boxes (list[torch.Tensor]): list of bounding boxes.
    
    Returns:
        replaced_images (list[torch.Tensor]): list of images with replaced regions.
    """
    
    replaced_images = []
    for image, new_region, region_mask, (x_min, y_min, x_max, y_max) in zip(images, new_regions, region_masks, bounding_boxes):
        replaced_img = image.clone()
        
        # apply mask again
        new_region = new_region * region_mask
        
        # mask out non-region area
        replaced_img[:, x_min: x_max + 1, y_min: y_max + 1] *= ~region_mask
        
        # add back new region
        replaced_img[:, x_min: x_max + 1, y_min: y_max + 1] += new_region
        
        replaced_images.append(replaced_img)
    
    return replaced_images
    
def cal_metrics(labels: torch.Tensor, preds: torch.Tensor, wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    Compute metrics (loss, accuracy, MCC score, precision, recall, F1 score) using ground truth labels and logits.

    Arguments:
        labels (torch.Tensor): ground truth labels.
        preds (torch.Tensor): logits (not softmaxed yet).
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """
    
    # loss
    loss = F.cross_entropy(preds, labels)
    wandb_log[metric_prefix + 'loss'] = loss
        
    # get probability
    preds = torch.softmax(preds, axis = 1)

    # ROC AUC
    # try:
    #     wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
    # except Exception:
    #     wandb_log[metric_prefix + 'auc'] = -1

    # get class prediction
    preds = preds.argmax(axis = 1)
    
    # accuracy and mcc
    for metric_name, metric_func in metrics_no_avg.items():
        metric = metric_func(labels, preds)
        wandb_log[metric_prefix + metric_name] = metric

    # precision, recall, f1 score
    # for metric_name, metric_func in metrics_with_avg.items():
    #     metric = metric_func(labels, preds, average = avg, zero_division = 0)
    #     wandb_log[metric_prefix + metric_name] = metric

def cal_IoUs(preds: torch.Tensor, targets: torch.Tensor, num_class: int = 4, eps: float = 1e-6) -> tuple[list, torch.Tensor]:
    """
    Calculate IoU per class and mean IoU. Reference: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy.

    Arguments:
        preds (torch.Tensor): predicted segmentation labels.
        targets (torch.Tensor): ground truth segmentation labels.
        num_class (int): number of unique classes in segmentation.
        eps (float): stabilizer.

    Returns:
        iou_per_class (list[torch.Tensor]): iou per class.
        miou (torch.Tensor): mean iou.
    """
 
    # preds and targets are of shape b * h * w
    iou_per_class = []
    
    for cls in range(num_class):
        pred_class = (preds == cls).float()
        true_class = (targets == cls).float()

        intersection = (pred_class * true_class).sum(dim=(1, 2))
        union = (pred_class + true_class).clamp(0, 1).sum(dim=(1, 2))

        iou = intersection / (union + eps)
        iou_per_class.append(iou)
    
    ious = torch.stack(iou_per_class, dim = 1)
    miou = ious.mean(dim = 1)

    return iou_per_class, miou

def angular_distance(v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Compute radian and degree distances between two normalized 3D vectors (i.e., gaze vectors).
    
    Arguments:
        v1 (torch.Tensor): tensor of shape (N, 3) - first set of 3D unit vectors, should be already normalized.
        v2 (torch.Tensor): tensor of shape (N, 3) - second set of 3D unit vectors, should be already normalized.
    
    Returns:
        radian (torch.Tensor): tensor of shape (N,) - radian distance.
        degree (torch.Tensor): tensor of shape (N,) - degree distance.
    """
    # compute dot product
    dot_product = torch.sum(v1 * v2, dim = 1)  # shape: (N,)

    # clamp to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # compute angle in radians
    radian = torch.acos(dot_product)  # shape: (N,)
    
    # convert to degree
    degree = torch.rad2deg(radian)

    return radian, degree

def GramMatrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized gram matrix for feature map.

    Arguments:
        x (torch.Tensor): feature map.

    Returns:
        x (torch.Tensor): normalized gram matrix.
    """

    x = x.flatten(start_dim = -2) # flatten w and h of feature map
    n = x[0].numel() # number of elements in gram matrix
    x = x @ x.transpose(-2, -1) 
    x = x / n # normalize gram matrix
    return x

class ContentLoss_L2(torch.nn.Module):
    """
    Loss function for MSE-based content loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target content features.
            weights (list[float]): weight for each layer.
        """
        
        super().__init__()
        self.targets = targets
        self.weights = [1.0] * len(targets) if weights is None else weights
        # self.weights = [w / sum(self.weights) for w in weights]   # normalize weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): content features of input image.

        Returns:
            loss (torch.Tensor): content loss.
        """
        
        loss = 0
        for p, t, w in zip(preds, self.targets, self.weights):
            # loss += ((p - t)**2).sum() * w
            loss += F.mse_loss(p, t) * w
        loss *= 0.5
        return loss
    
class StyleLoss_Gram(torch.nn.Module):
    """
    Loss function for MSE-based style loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target style features.
            weights (list[float]): weight for each layer.
        """
        
        super().__init__()
        self.targets = [GramMatrix(t) for t in targets]
        self.weights = [1.0] * len(targets) if weights is None else weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): style features of input image.

        Returns:
            loss (torch.Tensor): style loss.
        """

        loss = 0
        for p, t, w in zip(preds, self.targets, self.weights):
            p = GramMatrix(p)
            loss += ((p - t)**2).sum() * w
        loss *= 0.25
        return loss

class StyleLoss_BN(torch.nn.Module):
    """
    Loss function for BN statistics-based style loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target style features.
            weights (list[float]): weight for each layer.
        """

        super().__init__()
        self.targets_mean = [t.mean(dim = (-2, -1)) for t in targets]
        self.targets_std = [t.std(dim = (-2, -1)) for t in targets]
        self.weights = [1.0] * len(targets) if weights is None else weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): style features of input image.

        Returns:
            loss (torch.Tensor): style loss.
        """

        loss = 0
        for p, t_mean, t_std, w in zip(preds, self.targets_mean, self.targets_std, self.weights):
            p_mean = p.mean(dim = (-2, -1))
            p_std = p.std(dim = (-2, -1))
            loss += ((p_mean - t_mean)**2 + (p_std - t_std)**2).sum() * w / p_mean.shape[-1]
        return loss
    
def polar_transform(iris: torch.Tensor) -> tuple[np.ndarray, polarTransform.imageTransform.ImageTransform]:
    """
    Transform an image (iris) to polar coordinate. The iris image should be cropped with the help of segmentation mask.
    
    Arguments:
        iris (torch.Tensor): iris image tensor.
    
    Returns:
        TargetpolarImage (np.ndarray): polar image.
        ptSettingsTarget (polarTransform.imageTransform.ImageTransform): polar transformation settings.
    """
    
    assert(len(iris.shape) == 2 or (len(iris.shape) == 3 and iris.shape[0] == 1)) # iris should be of shape (h, w) or (1, h, w)    
    if len(iris.shape) == 3:
        iris = iris[0]
        
    TargetpolarImage, ptSettingsTarget = polarTransform.convertToPolarImage(iris)

    return TargetpolarImage, ptSettingsTarget