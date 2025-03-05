import tqdm
import torch
import wandb
import argparse
import torchvision.transforms.v2 as transforms

# self-defined functions
from models import VGG19, Classifier1, Classifier2, RITnet
from data_preprocessing import load_data_openeds2019, OpenEDS2019Dataset
from pipelines import iris_style_transfer, rubber_sheet, downsampling, gaussian_noise, gaussian_blur
from utils import seed, cal_metrics, cal_IoUs, prepare_dir, crop_image, mask_images, replace_image_region

def benchmark_openeds2019(
    args: argparse.Namespace,
    metric_prefix: str = 'test/',
    save_dir: str = 'saved/openeds2019/',
    ) -> None:
    """
    Main function for benchmarks on the OpenEDS2019 dataset.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        metric_prefix (str): wandb log prefix.
        save_dir (str): path to save directory.
    """
    
    # reproducibility
    seed(args.seed)

    # prepare folder for savings
    prepare_dir(save_dir)
    
    # load data (test set)
    test_x, test_y, test_m, num_class = load_data_openeds2019(test_split_ratio = args.test_split_ratio)
    dataset = OpenEDS2019Dataset(test_x , test_y , test_m , device = args.device)
    dataloader = torch.utils.data.DataLoader(dataset , batch_size = args.bs, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)
    print('number of classes:', num_class)
    
    # models
    vgg = VGG19() ; vgg.to(args.device)
    ritnet = RITnet() ; ritnet.to(args.device)
    classifier1 = Classifier1(num_class = num_class) ; classifier1.load_state_dict(torch.load(args.classifier1_path, weights_only = True, map_location = 'cpu')) ; classifier1.to(args.device) ; classifier1.eval()
    classifier2 = Classifier2(num_class = num_class) ; classifier2.load_state_dict(torch.load(args.classifier2_path, weights_only = True, map_location = 'cpu')) ; classifier2.to(args.device) ; classifier2.eval()

    # method names
    names = ['style_transfer', 'rubber_sheet', 'downsampling', 'gaussian_noise', 'gaussian_blur']
            
    # metrics lists and dicts
    c_preds1_pre, c_preds2_pre, c_labelss, s_labelss, ious0_pre, ious1_pre, ious2_pre, ious3_pre, mious_pre = [], [], [], [], [], [], [], [], []
    c_preds1_post, c_preds2_post, ious0_post, ious1_post, ious2_post, ious3_post, mious_post = {}, {}, {}, {}, {}, {}, {}
    for n in names:
        c_preds1_post[n] = []
        c_preds2_post[n] = []
        ious0_post[n] = []
        ious1_post[n] = []
        ious2_post[n] = []
        ious3_post[n] = []
        mious_post[n] = []
    
    # transforms
    t_resize = transforms.Resize((224, 224))
    t_toPIL = transforms.ToPILImage()
        
    for batch_id, (c_images, c_labels, c_masks_gt, s_images, s_labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            # save the 1st image of every batch
            t_toPIL(c_images[0]).save(save_dir + 'batch_' + str(batch_id) + '_raw.png')
            
            batch_wandb_log = {}
            c_labelss.append(c_labels)
            s_labelss.append(s_labels)

            # eye segmentation before manipulation
            c_masks_ritnet = [ritnet(c_img) for c_img in c_images]
            c_masks_ritnet = torch.stack(c_masks_ritnet).squeeze(1)
            
            # IoUs before manipulation
            iou_per_class, miou = cal_IoUs(c_masks_ritnet, c_masks_gt)
            ious0_pre.append(iou_per_class[0]) ; batch_wandb_log[metric_prefix + 'pre/batch/iou0'] = torch.nanmean(iou_per_class[0])
            ious1_pre.append(iou_per_class[1]) ; batch_wandb_log[metric_prefix + 'pre/batch/iou1'] = torch.nanmean(iou_per_class[1])
            ious2_pre.append(iou_per_class[2]) ; batch_wandb_log[metric_prefix + 'pre/batch/iou2'] = torch.nanmean(iou_per_class[2])
            ious3_pre.append(iou_per_class[3]) ; batch_wandb_log[metric_prefix + 'pre/batch/iou3'] = torch.nanmean(iou_per_class[3])
            mious_pre.append(miou) ; batch_wandb_log[metric_prefix + 'pre/batch/miou'] = torch.nanmean(miou)
                        
            # collect original iris regions ans their bounding boxes
            c_irises, c_iris_bbs, c_masks_ritnet_cropped = [], [], []
            for c_img, c_m_ritnet in zip(c_images, c_masks_ritnet):
                # apply mask and compute bounding box
                c_m_ritnet = c_m_ritnet == 2
                c_img = c_img * c_m_ritnet
                x_min, y_min, x_max, y_max = crop_image(c_img, return_idx = True)
                c_iris_bbs.append((x_min, y_min, x_max, y_max))
                
                # apply bounding box
                c_iris = c_img[:, x_min: x_max + 1, y_min: y_max + 1]
                c_irises.append(c_iris)
                c_m_ritnet_cropped = c_m_ritnet[..., x_min: x_max + 1, y_min: y_max + 1]
                c_masks_ritnet_cropped.append(c_m_ritnet_cropped)
            
            # classifications before manipulation
            c_irises_cnn = [t_resize(iris) for iris in c_irises]
            c_irises_cnn = torch.stack(c_irises_cnn)
            c_irises_cnn = c_irises_cnn.repeat(1, 3, 1, 1)
            x, x_c, x_s = vgg(c_irises_cnn)
            pred1 = classifier1(x)
            pred2 = classifier2(x_s)
            c_preds1_pre.append(pred1)
            c_preds2_pre.append(pred2)
            cal_metrics(c_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'pre/c1/batch/')
            cal_metrics(c_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'pre/c2/batch/')
            cal_metrics(s_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'pre/c1/mis/batch/')
            cal_metrics(s_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'pre/c2/mis/batch/')
            
            # collect attacker iris regions
            s_irises = []
            for s_img in s_images:
                # apply mask and apply bounding box
                s_m_ritnet = ritnet(s_img) == 2
                s_iris = s_img * s_m_ritnet
                s_iris = crop_image(s_iris)
                s_irises.append(s_iris)
        
        # apply iris manipulation methods
        new_c_irises_style_transfer = iris_style_transfer(c_irises, s_irises, args.c_loss_weight, args.s_loss_weight, args.nst_epochs, vgg, args.glint_threshold, args.device)
        new_c_irises_rubber_sheet = rubber_sheet(c_irises, s_irises, args.resize_threshold)
        new_c_irises_downsampling = downsampling(c_irises, args.downsampling_factor)
        new_c_irises_gaussian_noise = gaussian_noise(c_irises, args.noise_sigma, args.glint_threshold)
        new_c_irises_gaussian_blur = gaussian_blur(c_irises, args.blur_sigma, args.blur_kernel_size, args.glint_threshold)
        
        with torch.no_grad():
            # mask the new irises again
            new_c_irises_style_transfer = mask_images(new_c_irises_style_transfer, c_masks_ritnet_cropped)
            new_c_irises_rubber_sheet = mask_images(new_c_irises_rubber_sheet, c_masks_ritnet_cropped)
            new_c_irises_downsampling = mask_images(new_c_irises_downsampling, c_masks_ritnet_cropped)
            new_c_irises_gaussian_noise = mask_images(new_c_irises_gaussian_noise, c_masks_ritnet_cropped)
            new_c_irises_gaussian_blur = mask_images(new_c_irises_gaussian_blur, c_masks_ritnet_cropped)
            
            # classifications after manipulation
            irises_list = [new_c_irises_style_transfer, new_c_irises_rubber_sheet, new_c_irises_downsampling, new_c_irises_gaussian_noise, new_c_irises_gaussian_blur]
            for new_c_irises, n in zip(irises_list, names):
                # resize to 224x224
                new_c_irises = [t_resize(iris) for iris in new_c_irises]
                new_c_irises = torch.stack(new_c_irises)
                new_c_irises = new_c_irises.repeat(1, 3, 1, 1)
                    
                x, x_c, x_s = vgg(new_c_irises)
                pred1 = classifier1(x)
                pred2 = classifier2(x_s)
                
                c_preds1_post[n].append(pred1)
                c_preds2_post[n].append(pred2)
                cal_metrics(c_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + n + '/post/c1/batch/')
                cal_metrics(c_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + n + '/post/c2/batch/')
                cal_metrics(s_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + n + '/post/c1/mis/batch/')
                cal_metrics(s_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + n + '/post/c2/mis/batch/')
                
            # replace the old iris with the new iris
            new_c_images_style_transfer = replace_image_region(c_images, new_c_irises_style_transfer, c_masks_ritnet_cropped, c_iris_bbs)
            new_c_images_rubber_sheet = replace_image_region(c_images, new_c_irises_rubber_sheet, c_masks_ritnet_cropped, c_iris_bbs)
            new_c_images_downsampling = replace_image_region(c_images, new_c_irises_downsampling, c_masks_ritnet_cropped, c_iris_bbs)
            new_c_images_gaussian_noise = replace_image_region(c_images, new_c_irises_gaussian_noise, c_masks_ritnet_cropped, c_iris_bbs)
            new_c_images_gaussian_blur = replace_image_region(c_images, new_c_irises_gaussian_blur, c_masks_ritnet_cropped, c_iris_bbs)
            
            images_list = [new_c_images_style_transfer, new_c_images_rubber_sheet, new_c_images_downsampling, new_c_images_gaussian_noise, new_c_images_gaussian_blur]
            for new_c_imgs, n in zip(images_list, names):
                # save the 1st image of the batch
                t_toPIL(new_c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_' + n + '_new.png')
                
                # IoUs after manipulation
                c_masks_ritnet = [ritnet(c_img) for c_img in new_c_imgs]
                c_masks_ritnet = torch.stack(c_masks_ritnet).squeeze(1)
                iou_per_class, miou = cal_IoUs(c_masks_ritnet, c_masks_gt)
                
                ious0_post[n].append(iou_per_class[0]) ; batch_wandb_log[metric_prefix + n + '/post/batch/iou0'] = torch.nanmean(iou_per_class[0])
                ious1_post[n].append(iou_per_class[1]) ; batch_wandb_log[metric_prefix + n + '/post/batch/iou1'] = torch.nanmean(iou_per_class[1])
                ious2_post[n].append(iou_per_class[2]) ; batch_wandb_log[metric_prefix + n + '/post/batch/iou2'] = torch.nanmean(iou_per_class[2])
                ious3_post[n].append(iou_per_class[3]) ; batch_wandb_log[metric_prefix + n + '/post/batch/iou3'] = torch.nanmean(iou_per_class[3])
                mious_post[n].append(miou) ; batch_wandb_log[metric_prefix + n + '/post/batch/miou'] = torch.nanmean(miou)
            
        # batch log
        wandb.log(batch_wandb_log)

    # metrics
    wandb_log = {}
    with torch.no_grad():
        # save ious to files
        ious0_pre = torch.cat(ious0_pre) ; torch.save(ious0_pre, save_dir + 'ious0_pre.pt') ; wandb_log[metric_prefix + 'pre/mean_iou0'] = torch.nanmean(ious0_pre)
        ious1_pre = torch.cat(ious1_pre) ; torch.save(ious1_pre, save_dir + 'ious1_pre.pt') ; wandb_log[metric_prefix + 'pre/mean_iou1'] = torch.nanmean(ious1_pre)
        ious2_pre = torch.cat(ious2_pre) ; torch.save(ious2_pre, save_dir + 'ious2_pre.pt') ; wandb_log[metric_prefix + 'pre/mean_iou2'] = torch.nanmean(ious2_pre)
        ious3_pre = torch.cat(ious3_pre) ; torch.save(ious3_pre, save_dir + 'ious3_pre.pt') ; wandb_log[metric_prefix + 'pre/mean_iou3'] = torch.nanmean(ious3_pre)
        mious_pre = torch.cat(mious_pre) ; torch.save(mious_pre, save_dir + 'mious_pre.pt') ; wandb_log[metric_prefix + 'pre/mean_miou'] = torch.nanmean(mious_pre)
        
        for n in names:
            ious0_post[n] = torch.cat(ious0_post[n]) ; torch.save(ious0_post[n], save_dir + 'ious0_post_' + n + '.pt') ; wandb_log[metric_prefix + n + '/post/mean_iou0'] = torch.nanmean(ious0_post[n])
            ious1_post[n] = torch.cat(ious1_post[n]) ; torch.save(ious1_post[n], save_dir + 'ious1_post_' + n + '.pt') ; wandb_log[metric_prefix + n + '/post/mean_iou1'] = torch.nanmean(ious1_post[n])
            ious2_post[n] = torch.cat(ious2_post[n]) ; torch.save(ious2_post[n], save_dir + 'ious2_post_' + n + '.pt') ; wandb_log[metric_prefix + n + '/post/mean_iou2'] = torch.nanmean(ious2_post[n])
            ious3_post[n] = torch.cat(ious3_post[n]) ; torch.save(ious3_post[n], save_dir + 'ious3_post_' + n + '.pt') ; wandb_log[metric_prefix + n + '/post/mean_iou3'] = torch.nanmean(ious3_post[n])
            mious_post[n] = torch.cat(mious_post[n]) ; torch.save(mious_post[n], save_dir + 'mious_post_' + n + '.pt') ; wandb_log[metric_prefix + n + '/post/mean_miou'] = torch.nanmean(mious_post[n])
                
        # classification performance and false acceptance rate before iris manipulation
        c_labelss = torch.cat(c_labelss).cpu()
        s_labelss = torch.cat(s_labelss).cpu()
        cal_metrics(c_labelss, torch.cat(c_preds1_pre).cpu(), wandb_log, metric_prefix + 'pre/c1/' )
        cal_metrics(c_labelss, torch.cat(c_preds2_pre).cpu(), wandb_log, metric_prefix + 'pre/c2/' )
        cal_metrics(s_labelss, torch.cat(c_preds1_pre).cpu(), wandb_log, metric_prefix + 'pre/c1/mis/')
        cal_metrics(s_labelss, torch.cat(c_preds2_pre).cpu(), wandb_log, metric_prefix + 'pre/c2/mis/')
        
        # classification performance and false acceptance rate after iris manipulation
        for n in names:
            cal_metrics(c_labelss, torch.cat(c_preds1_post[n]).cpu(), wandb_log, metric_prefix + n + '/post/c1/')
            cal_metrics(c_labelss, torch.cat(c_preds2_post[n]).cpu(), wandb_log, metric_prefix + n + '/post/c2/')
            cal_metrics(s_labelss, torch.cat(c_preds1_post[n]).cpu(), wandb_log, metric_prefix + n + '/post/c1/mis/')
            cal_metrics(s_labelss, torch.cat(c_preds2_post[n]).cpu(), wandb_log, metric_prefix + n + '/post/c2/mis/')
        
    wandb.log(wandb_log)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-manipulation-benchmark', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-path1', '--classifier1_path', type = str, default = './models/weights/seed_42_Classifier1_lr_1e-05_prob_0.0_epoch_100.pth', help = 'pretrained classifier1 weight path')
    parser.add_argument('-path2', '--classifier2_path', type = str, default = './models/weights/seed_42_Classifier2_lr_1e-05_prob_0.0_epoch_500.pth', help = 'pretrained classifier2 weight path')
    parser.add_argument('-W', '--num_workers', type = int, default = 0, help = 'number of workers for data loader')
    parser.add_argument('-M', '--pin_memory', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to use pin memory for data loader')
    
    # hyperparameters for all iris manipulation methods
    parser.add_argument('-T', '--test_split_ratio', type = float, default = 0.2, help = 'train-test-split ratio')
    parser.add_argument('-bs', '--bs', type = int, default = 64, help = 'batch size')
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
    args.name = 'openeds2019 seed ' + str(args.seed)
    wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
    # benchmark main function
    benchmark_openeds2019(args)
    
    wandb.finish()