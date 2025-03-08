import tqdm
import torch
import wandb
import argparse
from PIL import Image
import torchvision.transforms.v2 as transforms

# self-defined functions
from data_preprocessing import load_data_openeds2020
from models import GazeEstimator1, GazeEstimator2, EfficientNet, VGG19
from utils import seed, angular_distance, prepare_dir, crop_image, replace_image_region
from pipelines import iris_style_transfer, rubber_sheet, downsampling, gaussian_noise, gaussian_blur

def benchmark_openeds2020(
    args: argparse.Namespace,
    metric_prefix: str = 'val/',
    save_dir: str = 'saved/openeds2020-ablation/',
    save_period: int = 25
    ) -> None:
    """
    Main function for benchmarks on the OpenEDS2020 dataset.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        metric_prefix (str): wandb log prefix.
        save_dir (str): path to save directory.
        save_period (int): save period.
    """
    
    # reproducibility
    seed(args.seed)
    
    # prepare folder for savings
    prepare_dir(save_dir)
    
    # load data (validation set, skipping traing set and test set because they are too large)
    c_images, labels = load_data_openeds2020(postfix = 'validation/')
    print('number of samples:', len(c_images))
    torch.save(labels, save_dir + 'gts.pt')
    
    # dataset and dataloader
    dataset = torch.utils.data.TensorDataset(c_images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.bs, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)        
        
    # models
    vgg = VGG19() ; vgg.to(args.device)
    efficientnet = EfficientNet() ; efficientnet.to(args.device)
    estimator1 = GazeEstimator1(extract_feature = True) ; estimator1.load_state_dict(torch.load(args.estimator1_path, weights_only = True, map_location = 'cpu')) ; estimator1.to(args.device) ; estimator1.eval()
    estimator2 = GazeEstimator2(extract_feature = True, freeze_resnet = False) ; estimator2.load_state_dict(torch.load(args.estimator2_path, weights_only = True, map_location = 'cpu')) ; estimator2.to(args.device) ; estimator2.eval()
    
    # transforms
    t_toPIL = transforms.ToPILImage()
    
    # a randomly chosen but fixed attacker image
    s_image = Image.open(args.attacker_path).convert('L')
    s_image = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])(s_image)
    s_image = s_image.to(args.device)
    t_toPIL(s_image).save(save_dir + 'sty.png')
    
    # extract iris from attacker image
    s_m_efficientnet = efficientnet(s_image)
    s_m_efficientnet = s_m_efficientnet == 2
    s_iris = s_image * s_m_efficientnet
    s_iris = crop_image(s_iris)
    t_toPIL(s_iris).save(save_dir + 'sty_iris.png')
    
    # method names
    # names = ['style_transfer', 'rubber_sheet', 'downsampling', 'gaussian_noise', 'gaussian_blur']
    names = ['gaussian_blur', 'gaussian_noise', 'downsampling', 'rubber_sheet', 'style_transfer']
    
    # metrics lists and dicts
    preds1_pre, preds2_pre, labelss = [], [], []
    preds1_post, preds2_post = {}, {}
    for n in names:
        preds1_post[n] = []
        preds2_post[n] = []
    
    for batch_id, (c_images, labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            c_images = c_images.to(args.device)
            
            # save the 1st image of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_images[0]).save(save_dir + 'batch_' + str(batch_id) + '_raw.png')
                
            batch_wandb_log = {}
            labelss.append(labels)
            
            # gaze estimation before manipulation
            c_segs = efficientnet(c_images)
            preds1 = estimator1(c_segs)
            preds2 = estimator2(c_images)
            preds1_pre.append(preds1)
            preds2_pre.append(preds2)
            radian_distances1, degree_distances1 = angular_distance(preds1.cpu(), labels)
            radian_distances2, degree_distances2 = angular_distance(preds2.cpu(), labels)
            batch_wandb_log[metric_prefix + 'pre/batch/radian_distance1'] = radian_distances1.mean()
            batch_wandb_log[metric_prefix + 'pre/batch/degree_distance1'] = degree_distances1.mean()
            batch_wandb_log[metric_prefix + 'pre/batch/radian_distance2'] = radian_distances2.mean()
            batch_wandb_log[metric_prefix + 'pre/batch/degree_distance2'] = degree_distances2.mean()
            
            # collect original iris regions ans their bounding boxes
            c_irises, c_iris_bbs, c_masks_efficientnet_cropped = [], [], []
            for c_img, c_seg in zip(c_images, c_segs):
                # apply mask and compute bounding box
                c_m_efficientnet = c_seg == 2
                c_img = c_img * c_m_efficientnet
                
                c_m_glint = (c_img <= args.glint_threshold) * c_m_efficientnet
                c_img2 = c_img * c_m_glint
                
                x_min, y_min, x_max, y_max = crop_image(c_img2, return_idx = True)
                c_iris_bbs.append((x_min, y_min, x_max, y_max))
                
                # apply bounding box
                c_iris = c_img[:, x_min: x_max + 1, y_min: y_max + 1]
                c_irises.append(c_iris)
                
                # crop mask
                c_m_efficientnet_cropped = c_m_efficientnet[..., x_min: x_max + 1, y_min: y_max + 1]
                c_masks_efficientnet_cropped.append(c_m_efficientnet_cropped)
                
            # save the 1st iris of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_irises[0]).save(save_dir + 'batch_' + str(batch_id) + '_raw_iris.png')
        
        # apply iris manipulation methods
        new_c_irises_gaussian_blur = gaussian_blur(c_irises, args.blur_sigma, args.blur_kernel_size, args.glint_threshold)
        new_c_irises_gaussian_noise = gaussian_noise(c_irises, args.noise_sigma, args.glint_threshold)
        new_c_irises_downsampling = downsampling(c_irises, args.downsampling_factor)
        new_c_irises_rubber_sheet = rubber_sheet(c_irises, [s_iris], args.resize_threshold)
        new_c_irises_style_transfer = iris_style_transfer(c_irises, [s_iris], args.c_loss_weight, args.s_loss_weight, args.nst_epochs, vgg, args.glint_threshold, args.device)
        
        with torch.no_grad():
            # mask the new irises again
            irises_list = [new_c_irises_gaussian_blur, new_c_irises_gaussian_noise, new_c_irises_downsampling, new_c_irises_rubber_sheet, new_c_irises_style_transfer]
            
            # mask the new irises again
            for i in range(len(irises_list)):
                irises_list[i] = [iris * m for iris, m in zip(irises_list[i], c_masks_efficientnet_cropped)]
                
                # save the 1st iris of the batch
                if batch_id % save_period == 0:
                    t_toPIL(irises_list[i][0]).save(save_dir + 'batch_' + str(batch_id) + '_' + names[i] + '_new_iris.png')
            
            # replace the old iris with the new iris
            images_list = [replace_image_region(c_images, new_c_irises, c_iris_bbs, c_masks_efficientnet_cropped) for new_c_irises in irises_list]
            for new_c_imgs, n in zip(images_list, names):
                # save the 1st image of the batch
                if batch_id % save_period == 0:
                    t_toPIL(new_c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_' + n + '_new.png')
                
                new_c_imgs = torch.stack(new_c_imgs)
                new_c_imgs = new_c_imgs.repeat(1, 3, 1, 1)
                
                # gaze estimation after manipulation
                c_segs = efficientnet(new_c_imgs)
                preds1 = estimator1(c_segs)
                preds2 = estimator2(new_c_imgs)
                
                preds1_post[n].append(preds1)
                preds2_post[n].append(preds2)
            
                radian_distances1, degree_distances1 = angular_distance(preds1.cpu(), labels)
                radian_distances2, degree_distances2 = angular_distance(preds2.cpu(), labels)
                batch_wandb_log[metric_prefix + n + '/post/batch/radian_distance1'] = radian_distances1.mean()
                batch_wandb_log[metric_prefix + n + '/post/batch/degree_distance1'] = degree_distances1.mean()
                batch_wandb_log[metric_prefix + n + '/post/batch/radian_distance2'] = radian_distances2.mean()
                batch_wandb_log[metric_prefix + n + '/post/batch/degree_distance2'] = degree_distances2.mean()

        # batch log
        wandb.log(batch_wandb_log)

    # metrics
    wandb_log = {}
    with torch.no_grad():
        # concatenate all tensors, save to files, compute metrics
        labelss = torch.cat(labelss).detach().cpu() ; torch.save(labelss, save_dir + 'labels.pt')
        preds1_pre = torch.cat(preds1_pre).detach().cpu() ; torch.save(preds1_pre, save_dir + 'preds1_pre.pt')
        preds2_pre = torch.cat(preds2_pre).detach().cpu() ; torch.save(preds2_pre, save_dir + 'preds2_pre.pt')
        
        radian_distances1, degree_distances1 = angular_distance(preds1_pre, labelss)
        radian_distances2, degree_distances2 = angular_distance(preds2_pre, labelss)
        wandb_log[metric_prefix + 'pre/radian_distance1'] = radian_distances1.mean()
        wandb_log[metric_prefix + 'pre/degree_distance1'] = degree_distances1.mean()
        wandb_log[metric_prefix + 'pre/radian_distance2'] = radian_distances2.mean()
        wandb_log[metric_prefix + 'pre/degree_distance2'] = degree_distances2.mean()
        
        for n in names:
            preds1_post[n] = torch.cat(preds1_post[n]).detach().cpu() ; torch.save(preds1_post[n], save_dir + 'preds1_post_' + n + '.pt')
            preds2_post[n] = torch.cat(preds2_post[n]).detach().cpu() ; torch.save(preds2_post[n], save_dir + 'preds2_post_' + n + '.pt')
        
            radian_distances1, degree_distances1 = angular_distance(preds1_post[n], labelss)
            radian_distances2, degree_distances2 = angular_distance(preds2_post[n], labelss)
            wandb_log[metric_prefix + n + '/post/radian_distance1'] = radian_distances1.mean()
            wandb_log[metric_prefix + n + '/post/degree_distance1'] = degree_distances1.mean()
            wandb_log[metric_prefix + n + '/post/radian_distance2'] = radian_distances2.mean()
            wandb_log[metric_prefix + n + '/post/degree_distance2'] = degree_distances2.mean()        
        
    wandb.log(wandb_log)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
        
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-manipulation-benchmark', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-path1', '--estimator1_path', type = str, default = './models/weights/seed_42_GazeEstimator1_lr_1e-05_epoch_500.pth', help = 'pretrained estimator1 weight path')
    parser.add_argument('-path2', '--estimator2_path', type = str, default = './models/weights/seed_42_GazeEstimator2_lr_1e-05_epoch_150.pth', help = 'pretrained estimator2 weight path')
    parser.add_argument('--attacker_path', type = str, default = '../data/openeds2020/openEDS2020-GazePrediction/test/sequences/2577/023.png', help = 'attacker image path')
    parser.add_argument('-W', '--num_workers', type = int, default = 0, help = 'number of workers for data loader')
    parser.add_argument('-M', '--pin_memory', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to use pin memory for data loader')
    
    # hyperparameters for all iris manipulation methods
    parser.add_argument('-T', '--test_split_ratio', type = float, default = 0.2, help = 'train-test-split ratio')
    parser.add_argument('-bs', '--bs', type = int, default = 128, help = 'batch size')
    parser.add_argument('--glint_threshold', type = float, default = 0.8, help = 'glint threshold')
    
    # for iris style transfer
    parser.add_argument('-E', '--nst_epochs', type = int, default = 200, help = 'number of epochs for neural style transfer')
    parser.add_argument('-cw', '--c_loss_weight', type = int, default = 1, help = 'cw')
    parser.add_argument('-sw', '--s_loss_weight', type = int, default = 1, help = 'sw')
    
    # for rubber sheet
    parser.add_argument('--resize_threshold', type = float, default = 0.1, help = 'resize threshold')
    
    # for downsampling
    parser.add_argument('--downsampling_factor', type = float, default = 2, help = 'downsampling factor')

    # for gaussian noise
    parser.add_argument('--noise_sigma', type = float, default = 0.05, help = 'noise std')    
    
    # for gaussian blur
    parser.add_argument('--blur_sigma', type = float, default = 2.0, help = 'blur std')
    parser.add_argument('--blur_kernel_size', type = int, default = None, help = 'blur kernel size')
        
    args = parser.parse_args()
    args.device = 'cuda:' + str(args.device) if args.device >= 0 else 'cpu'
    
    # wandb init
    args.name = 'openeds2020 ablation seed ' + str(args.seed)
    wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
    # benchmark main function
    benchmark_openeds2020(args)
    
    wandb.finish()