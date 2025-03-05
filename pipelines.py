import cv2
import tqdm
import torch
import numpy as np
import torchvision.transforms.v2 as transforms

# self-defined functions
from models import VGG19
from utils import ContentLoss_L2, StyleLoss_BN, StyleLoss_Gram, polar_transform

def nst(c_img: torch.Tensor, 
        s_img: torch.Tensor, 
        clone_content: bool = True, 
        BN_loss: bool = True,
        c_loss_weight: float = 1,
        s_loss_weight: float = 1,
        lr: float = 1,
        epochs: int = 200,
        vgg: torch.nn.Module = None,
        use_tqdm: bool = True,
        device: str = 'cuda:0',
        ) ->  tuple[torch.Tensor, list, list, list]:
    """
    Neural style transfer pipeline.

    Arguments:
        c_img (torch.Tensor): content image tensor. 
        s_img (torch.Tensor): style image tensor.
        clone_content (bool): whether the initilization is a clone on content image or random noise. 
        BN_loss (bool): whether to use BN style loss or gram matrix style loss.
        c_loss_weight (float): content loss weight (alpha).
        s_loss_weight (float): style loss weight (beta).
        lr (float): learning rate.
        epochs (int): style transfer iterations.
        vgg (torch.nn.Module): vgg model.
        use_tqdm (bool): whether to use tqdm for showing style transfer progress.
        device (str): CPU or GPU.
        
    Returns:
        x (torch.Tensor): stylized image tensor.
        x_hist (list[torch.Tensor]): stylized image tensor history. 
        c_loss_hist (list[float]): content loss history.
        s_loss_hist (list[float]): style loss history.
    """
    
    # vgg model
    if vgg is None:
        vgg = VGG19()
    vgg.to(device)

    # input and output images
    c_img = c_img.to(device)
    s_img = s_img.to(device)
    if clone_content:
        x = c_img.clone()
    else:
        x = torch.rand(c_img.shape).to(device)
    x = x.contiguous() # sometimes x is not contiguous for some unknown reason...
    x.requires_grad = True
    
    # optimizer
    optim = torch.optim.LBFGS([x], lr = lr)
    
    # content and style loss functions
    _, c_features, _ = vgg(c_img)
    _, _, s_features = vgg(s_img)
    c_loss_func = ContentLoss_L2(targets = c_features)
    if BN_loss:
        s_loss_func = StyleLoss_BN(targets = s_features)
    else:
        s_loss_func = StyleLoss_Gram(targets = s_features)
    
    # style transfer main loop
    x_hist = []
    c_loss_hist = []
    s_loss_hist = []
    current_epoch = [0]
    
    # nst loop
    if use_tqdm:
        pbar = tqdm.tqdm(total = epochs)
    while current_epoch[0] < epochs:
        def closure():
            with torch.no_grad():
                x.clamp_(0, 1)
                # x.mul_(mask)
                
            optim.zero_grad()
            _, x_c, x_s = vgg(x)
            c_loss = c_loss_func(x_c)
            s_loss = s_loss_func(x_s)
            loss = c_loss * c_loss_weight + s_loss * s_loss_weight
            loss.backward()

            # records
            x_hist.append(x.detach().cpu())
            c_loss_hist.append(c_loss.item())
            s_loss_hist.append(s_loss.item())
            
            current_epoch[0] += 1
            if use_tqdm:
                pbar.update(1)

            return loss
            
        optim.step(closure)

    if use_tqdm:
        pbar.close()

    x = x.detach()
    x.clamp_(0, 1)
    return x, x_hist, c_loss_hist, s_loss_hist

def iris_style_transfer(irises1: list[torch.Tensor], 
                        irises2: list[torch.Tensor],
                        c_loss_weight: float,
                        s_loss_weight: float,
                        epochs: int,
                        vgg: torch.nn.Module,
                        glint_threshold: float,
                        device: str,
                        ) -> list[torch.Tensor]:
    """
    Transfer the style of one iris region (irises2) to another (irises1). The input iris images should be cropped with the help of segmentation mask.
    
    Aguements:
        irises1 (list[torch.Tensor]): iris region tensors.
        irises2 (list[torch.Tensor]): iris region tensors.
        c_loss_weight (float): content loss weight.
        s_loss_weight (float): style loss weight.
        epochs (int): style transfer iterations.
        vgg (torch.nn.Module): vgg model.
        glint_threshold (float): threshold for glint removal.
        device (str): CPU or GPU.
    
    Returns:
        irises1_stylized (list[torch.Tensor]): stylized iris region tensors.
    """
    
    # remove glints
    glints1 = [iris * (iris > glint_threshold) for iris in irises1]
    irises1 = [iris * (iris <= glint_threshold) for iris in irises1]    
    irises2 = [iris * (iris <= glint_threshold) for iris in irises2]
    
    # record iris shapes
    irises1_shapes = [iris.shape[-2:] for iris in irises1]
    
    # resize to (224, 224)
    t_resize = transforms.Resize((224, 224))
    irises1 = [t_resize(iris) for iris in irises1] ; irises1 = torch.stack(irises1) ; irises1 = irises1.repeat(1, 3, 1, 1)
    irises2 = [t_resize(iris) for iris in irises2] ; irises2 = torch.stack(irises2) ; irises2 = irises2.repeat(1, 3, 1, 1)
    
    irises1_stylized, _, _, _ = nst(irises1, # content image
                                    irises2, # style image
                                    c_loss_weight = c_loss_weight, 
                                    s_loss_weight = s_loss_weight, 
                                    epochs = epochs, 
                                    vgg = vgg, 
                                    use_tqdm = False, 
                                    device = device)
    
    # RGB to grayscale
    irises1_stylized = transforms.functional.rgb_to_grayscale(irises1_stylized)
    
    # resize new iris texture to its original size
    irises1_stylized = [transforms.Resize(shape)(iris) for iris, shape in zip(irises1_stylized, irises1_shapes)]
    
    # add back glints
    irises1_stylized = [iris * (glint == 0) + glint for iris, glint in zip(irises1_stylized, glints1)]
    
    return irises1_stylized

def rubber_sheet(irises1: list[torch.Tensor], irises2: list[torch.Tensor], resize_threshold: float = 0.1) -> list[torch.Tensor]:
    """
    Fit the shape of one iris region (iris2) to another (iris1). The input iris images should be cropped with the help of segmentation mask.
    
    Arguments:
        irises1 (list[torch.Tensor]): iris region tensors.
        irises2 (list[torch.Tensor]): iris region tensors.
        threshold (float): threshold for compensation mask.
    
    Returns:
        irises2_fitted (list[torch.Tensor]): fitted iris region tensors.
    """
    
    device = irises1[0].device
    
    # irises2 may contain only one element
    if len(irises2) == 1:
        irises2 = irises2 * len(irises1)
    
    irises2_fitted = []
    for iris1, iris2 in zip(irises1, irises2):
        # polar transform
        iris1, setting1 = polar_transform(iris1.cpu())
        iris2, setting2 = polar_transform(iris2.cpu())
        
        # resize
        iris2_resized = cv2.resize(iris2, (iris1.shape[-1], iris1.shape[-2]), interpolation = cv2.INTER_CUBIC)
        
        # output
        iris2_fitted = iris1.copy()
        
        # masks for compensating resize interpolation
        mask1 = iris1 >= resize_threshold
        mask2 = iris2_resized >= resize_threshold

        # iterate over rows in iris region
        for j in range(iris2_resized.shape[0]):
            # skip empty rows
            if np.sum(mask1[j]) <= 0 or np.sum(mask2[j]) <= 0:
                # print('empty row', j)
                continue
            
            # find nonzero elements
            loc1 = np.nonzero(mask1[j])
            loc2 = np.nonzero(mask2[j])
            
            # nonzero regions
            left1, right1 = np.min(loc1), np.max(loc1)  
            left2, right2 = np.min(loc2), np.max(loc2)
            
            # width of nonzero region in iris1
            width1 = right1 - left1 + 1

            # resize rows in iris2_resized to match the detected iris width in iris1
            fitted = cv2.resize(iris2_resized[j][left2 : right2 + 1], (1, width1) ,interpolation = cv2.INTER_CUBIC)
            fitted = np.reshape(fitted, (width1,))
            # assert(len(fitted) == width1)
            
            # replace the rows in iris1 with the resized rows in iris2
            iris2_fitted[j, left1 : right1 + 1] = fitted
        
        # transform back to cartesian image   
        iris2_fitted = setting1.convertToCartesianImage(iris2_fitted)
        iris2_fitted = torch.tensor(iris2_fitted).to(device).unsqueeze(0)
        irises2_fitted.append(iris2_fitted)
        
    return irises2_fitted

def downsampling(irises: list[torch.Tensor], factor: int | float = 2) -> list[torch.Tensor]:
    """
    Downsample iris region tensors and then upsample them back to the original sizes.
    
    Arguments:
        irises (list[torch.Tensor]): iris region tensors.
        factor (int): downsampling factor.
        
    Returns:
        irises_downsampled (list[torch.Tensor]): downsampled iris region tensors.
    """
    
    irises_downsampled = []
    
    for iris in irises:
        t_down = transforms.Resize((int(iris.shape[-2] / factor), int(iris.shape[-1] / factor)))
        t_up = transforms.Resize(iris.shape[-2:])
        
        iris = t_down(iris)
        iris = t_up(iris)
        
        irises_downsampled.append(iris)
    
    return irises_downsampled

def gaussian_noise(irises: list[torch.Tensor], sigma: float = 0.05, glint_threshold: float = 0.8) -> list[torch.Tensor]:
    """
    Add Gaussian noise to iris region tensors.
    
    Arguments:
        irises (list[torch.Tensor]): iris region tensors.
        sigma (float): standard deviation of Gaussian noise.
        glint_threshold (float): threshold for glint removal.
        
    Returns:
        irises_noisy (list[torch.Tensor]): noisy iris region tensors.
    """
    
    
    glints = [iris * (iris > glint_threshold) for iris in irises]
    
    t_noise = transforms.GaussianNoise(sigma = sigma, clip = True)
    irises_noisy = t_noise(irises)
    
    # add back glints
    irises_noisy = [iris * (glint == 0) + glint for iris, glint in zip(irises_noisy, glints)]
    
    return irises_noisy

def gaussian_blur(irises: list[torch.Tensor], sigma: float = 2.0, kernel_size: int = None, glint_threshold: float = 0.8) -> list[torch.Tensor]:
    """
    Blur iris region tensors with Gaussian kernel.
    
    Arguments:
        irises (list[torch.Tensor]): iris region tensors.
        sigma (float): standard deviation of Gaussian kernel.
        kernel_size (int): kernel size of Gaussian kernel.
        glint_threshold (float): threshold for glint removal.
        
    Returns:
        irises_blurred (list[torch.Tensor]): blurred iris region tensors.
    """
    
    glints = [iris * (iris > glint_threshold) for iris in irises]
    irises = [iris * (iris <= glint_threshold) for iris in irises]
    
    # rule of thumb for kernel size
    if kernel_size is None:
        kernel_size = 6 * int(sigma) + 1 
    
    t_blur = transforms.GaussianBlur(kernel_size = kernel_size, sigma = sigma)
    irises_blurred = t_blur(irises)
    
    # add back glints
    irises_blurred = [iris * (glint == 0) + glint for iris, glint in zip(irises_blurred, glints)]
    
    return irises_blurred