from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import cv2
import numpy as np
import json
import re
import functools
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, ACDC, Kvasir
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'acdc','kvasir'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    
    parser.add_argument('--calculate_metrics' , action='store_true', default=False)

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    available_models.append('ensemble')

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')

    parser.add_argument("--num_models", type=int, default=5,
                        help='number of models in the ensemble (default: 5)')

    parser.add_argument("--backbone", type=str, default='deeplabv3plus_resnet50', choices=available_models,
                        help='backbone to use in the models of the ensemble (default: deeplabv3plus_resnet50)')

    parser.add_argument("--weight", type=str, action='append', default=None, help='weight for a model of the ensemble')

    parser.add_argument("--member", type=str, action='append', default=None, help='member model of the ensemble')

    parser.add_argument("--ensemble_lr", type=float, action='append', default=None, help='lr for each model in the ensemble')

    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--dropout",type=float, default=None,
                        help='dropout rate (default: None)')
    parser.add_argument("--alpha",type=float, default=0,
                        help='alpha parameter for uce loss (only use with uce loss, default: 0)')

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'uncertainty_cross_entropy', 'uce', 'dice', 'hybrid'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="manual random seed (default: 1). If set to -1, completely random seed.")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--make_uncertainty_masks",action='store_true',default=False)

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
             et.ExtResize(opts.crop_size),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            #et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            #et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'acdc':
        
        train_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            #et.ExtRandomScale((0.5,2.0)),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            #et.ExtRandomHorizontalFlip(0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        train_dst = ACDC(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = ACDC(root=opts.data_root,
                             split='val', transform=val_transform)
    
    if opts.dataset == 'kvasir':
        
        train_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtRandomScale((0.5,2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.55713434, 0.32152654, 0.23528269],
                            std=[0.30677355, 0.21518466, 0.1778166]),
            ])

        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.55713434, 0.32152654, 0.23528269],
                            std=[0.30677355, 0.21518466, 0.1778166]),
            ])

        train_dst = Kvasir(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Kvasir(root=opts.data_root,
                             split='val', transform=val_transform)

    #print(train_dst.__getitem__(5))

    return train_dst, val_dst

def make_uncertainty_masks(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            labels,uncertainty_masks = labels[0],labels[1]

            img_name = images[1][0].split('/')[-1][:-4]
            print(f'IMAGE NAME: {img_name}')
            images = images[0].to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)


            outputs = model(images)
            #uncertainty = utils.entropy(outputs).detach().cpu().numpy()
            #uncertainty = uncertainty.squeeze()
            #print(outputs.shape)
            probs = outputs
            log_probs = torch.log(probs+1e-10)
            entropy = -torch.sum(probs * log_probs, dim=1)
            uncertainty = entropy.detach().cpu().numpy()
            uncertainty = uncertainty.squeeze()

            #print(probs.shape)


            uncertainty_mask = create_instance_uncertainty_mask(probs,uncertainty,opts.num_classes,opts.crop_size)
            
            np.save(f'datasets/data/{opts.dataset}/uncertainty_masks/{img_name}_uncertainty_mask',uncertainty_mask)
            
    return


def create_instance_uncertainty_mask(logits_seg_mask,pixel_uncertainty_mask_np,num_classes,crop_size):
    preds_seg_mask = torch.argmax(torch.squeeze(logits_seg_mask),dim=0)
    preds_seg_mask_np = preds_seg_mask.detach().cpu().numpy()

    #print(preds_seg_mask_np)

	#resulting uncertainty mask
    instance_uncertainty_mask = torch.zeros(crop_size,crop_size)
    instance_uncertainty_mask_np = instance_uncertainty_mask.numpy()

    for k in range(num_classes):
        
        #Create a binary mask where the elements equal to k are set to 1, and others to 0
        binary_mask = (preds_seg_mask_np == k).astype(np.uint8)

        #print(preds_seg_mask_np)

        #print(binary_mask)

        # Use cv2.connectedComponents to find the connected components
        #print(binary_mask.shape)
        num_labels, labels = cv2.connectedComponents(binary_mask)

        # For each connected component, compute its mean uncertainty and store it in the resulting uncertainty mask
        for i in range(1,num_labels):
            instance_indices = np.where(labels == i)
            uncertainty_instance_values = pixel_uncertainty_mask_np[instance_indices]
            uncertainty_instance_mean = uncertainty_instance_values.mean()
            instance_uncertainty_mask_np[instance_indices] = uncertainty_instance_mean 

    return instance_uncertainty_mask_np

def validate_ensemble(opts,models,loader,device,metrics):
    metrics.reset()

    with torch.no_grad():
        for i,(images,labels) in tqdm(enumerate(loader)):
            labels, uncertainty_masks = labels[0],labels[1]

            images = images[0].to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = []
            for model in models:
                outputs.append(model(images))

            outputs_softmax = [F.softmax(output,dim=1) for output in outputs]

            ensemble_out = torch.mean(torch.stack(outputs_softmax),dim=0)

            preds = ensemble_out.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets,preds)

        score = metrics.get_results()

    return score, []

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    mpavpu = 0
    m_unc = 0

    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            labels,uncertainty_masks = labels[0],labels[1]

            #print(labels)

            images = images[0].to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            #print(torch.sum(outputs,dim=1))
            #mpavpu += utils.pavpu(outputs,labels,4,1,0.1398,0.5)/406
            mpavpu = 0

            #m_unc += torch.sum(utils.entropy(outputs).detach().cpu().squeeze())/(768*768)
            #print(outputs)

            # Apply sigmoid to the output to get probabilities between 0 and 1
            #preds_prob = torch.sigmoid(outputs.detach())

            # Threshold at 0.5 to get binary predictions (1 for foreground, 0 for background)
            #preds = (preds_prob > 0.5).long()  # Convert probabilities to binary values

            # Remove the extra channel dimension and move to CPU for metric calculation
            #preds = preds.squeeze(1).cpu().numpy()  # Convert to NumPy for metric computation

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            #print(np.unique(targets))

            #print(preds.shape)
            #print(targets.shape)

            #print(np.unique(targets))

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    
    with open("pavpu_resnet101.txt","a") as fp:
        fp.write(str(mpavpu)+'\n')

    return score, ret_samples

def _instantiate_and_set_bn_momentum(opts,backbone):
    print(opts.dropout)
    model = network.modeling.__dict__[backbone](num_classes=opts.num_classes, output_stride=opts.output_stride,dropout=opts.dropout)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    return model

def get_backbone(weight):
    weight = weight.split('/')[-1]
    args = weight.split('_')
    model = args[1]
    backbone = args[2]
    if args[3].isnumeric():
        return model+'_'+backbone+'_'+args[3]

    return model+'_'+backbone

def calculate_metrics(path,models):
    res = {}
    means = {}
    std_devs = {}
    
    for model in models:
        res[model] = []
        for file_path in os.listdir(path):
            #print(file_path)
            if re.match('[0-9]+' + r'_class_iou.json' ,file_path):
                #print(file_path)
                with open(os.path.join(path,file_path),'r') as open_fp:
                    res[model].append(json.load(open_fp))
        
        keys = res[model][0].keys()
        tmp = {}
        for m in res[model]:
            for k in keys:
                if k not in tmp:
                    tmp[k] = []
                tmp[k].append(m.get(k))

        means[model] = {}
        std_devs[model] = {}

        for k,v in tmp.items():
            array_vals = np.array(v)
            means[model][k] = round(np.mean(array_vals),3)
            means[model]['mIoU'] = round(np.mean(list(means[model].values())),3)
            std_devs[model][k] = round(np.std(array_vals),3)
    
    print(means)
    print(std_devs)
    return

    



def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'acdc':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'kvasir':
        opts.num_classes = 2
    
    if opts.calculate_metrics:
        calculate_metrics('./checkpoints/exp_paper/',['ensemble'])
        return

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    if opts.random_seed == -1:
        opts.random_seed = np.random.randint(1,10000000)
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are constructed at network.modeling)
    if opts.model == 'deep_ensemble':
        models = [ _instantiate_and_set_bn_momentum(opts,get_backbone(opts.weight[i])) for i in range(len(opts.weight))]
        model = network.modeling.deep_ensemble(models,opts.num_classes,opts.output_stride)

    elif opts.model == 'ensemble':
        models = [network.modeling.__dict__[opts.member[i]](num_classes=opts.num_classes, output_stride=opts.output_stride,dropout=opts.dropout) for i in range(len(opts.member))]
        for model in models:
            utils.set_bn_momentum(model.backbone, momentum=0.01)

    else:
        print(opts.dropout)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,dropout=opts.dropout)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    #print(opts.make_uncertainty_masks)

    if opts.model == 'ensemble':
        optimizers = [torch.optim.Adam(params=[
            {'params': models[i].backbone.parameters(), 'lr': 0.1 * opts.ensemble_lr[i]},{'params': models[i].classifier.parameters(), 'lr': opts.ensemble_lr[i]},], lr=opts.ensemble_lr[i], weight_decay=opts.weight_decay) for i in range(len(models))]

        schedulers = [utils.PolyLR(optimizer, opts.total_itrs, power=0.9) for optimizer in optimizers]

        criterions = [utils.loss.UncertaintyAwareCrossEntropyLoss(ignore_index=255,reduction='mean') for model in models]

    if not opts.test_only and not opts.make_uncertainty_masks and opts.model != 'ensemble':
        optimizer = torch.optim.Adam(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, weight_decay=opts.weight_decay)

        '''optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)'''
        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    print(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)

    elif opts.loss_type == 'cross_entropy':
        #criterion = nn.BCEWithLogitsLoss(reduction='mean')
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    elif opts.loss_type== 'uncertainty_cross_entropy':
        criterion = utils.loss.UncertaintyAwareCrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type== 'uce':
        criterion = utils.loss.Uce(ignore_index=255,reduction='mean',alpha=opts.alpha)
    elif opts.loss_type=='dice':
        criterion = utils.loss.DiceLoss()
    elif opts.loss_type == 'hybrid':
        criterion = utils.loss.HybridLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    def _save_ckpt(path,index):
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": models[index].module.state_dict(),
            "optimizer_state": optimizers[index].state_dict(),
            "scheduler_state": schedulers[index].state_dict(),
            "best_score": best_score[index],
            }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0
    cur_itrs = 0
    cur_epochs = 0

    if opts.weight is not None:
        for i in range(len(opts.weight)):
            checkpoint = torch.load(opts.weight[i],map_location=torch.device('cpu'))
            model.models[i].load_state_dict(checkpoint["model_state"])
            model.models[i] = nn.DataParallel(model.models[i])
            model.models[i].to(device)
            model.models[i].eval()
        model = nn.DataParallel(model)
        model.to(device)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory

    if opts.model == 'ensemble':
        for i in range(len(models)):
            models[i] = nn.DataParallel(models[i])
            models[i] = models[i].to(device)

    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.model=='ensemble':
        print('ok')
        
        interval_loss = [0 for i in range(len(models))]
        miou_history = []
        loss_history = []
        best_score_ensemble = 0
        losses = [0 for i in range(len(models))]
        best_score = [0 for i in range(len(models))]


        while True:
            cur_epochs +=1

            for (images, labels) in train_loader:
                cur_itrs += 1
                ensemble_outputs = []

                labels, loaded_uncertainty_masks = labels[0], labels[1]

                images = images[0].to(device, dtype = torch.float32)
                labels = labels.to(device, dtype=torch.long)

                for model in models:
                    model.train()
                    outputs = model(images)
                    ensemble_outputs.append(outputs)

                ensemble_outputs_softmax = [F.softmax(output,dim=1) for output in ensemble_outputs]
                ensemble_out = torch.mean(torch.stack(ensemble_outputs_softmax),dim=0)

                pixel_uncertainty = -torch.sum(ensemble_out * torch.log(ensemble_out + 1e-10), dim=1)
                pixel_uncertainty = pixel_uncertainty.detach().cpu().numpy()
                instance_uncertainty = []

                for i in range(pixel_uncertainty.shape[0]):
                    instance_uncertainty.append(create_instance_uncertainty_mask(ensemble_out[i],pixel_uncertainty[i],opts.num_classes,opts.crop_size))
                    
                instance_uncertainty = torch.tensor(np.array(instance_uncertainty))
                instance_uncertainty = instance_uncertainty.to(device,dtype=torch.float32)

                for i in range(len(models)):
                    optimizers[i].zero_grad()      

                    loss = criterions[i](ensemble_outputs[i],labels,instance_uncertainty)
                    np_loss = loss.detach().cpu().numpy()
                    losses[i] = np_loss
                    interval_loss[i] += losses[i] 
                    loss.backward()
                    optimizers[i].step()
                

                if (cur_itrs) % 10 == 0:
                    interval_loss = [l/10 for l in interval_loss]
                    print("Epoch %d, Itrs %d/%d" %
                        (cur_epochs, cur_itrs, opts.total_itrs))
                    for i in range(len(interval_loss)):

                        print(f'Model {opts.member[i]} Loss: {interval_loss[i]}')

                    interval_loss = [0 for l in interval_loss]

                if (cur_itrs) % opts.val_interval == 0:
                    if opts.loss_type != 'uncertainty_cross_entropy':
                        if opts.dropout != None:
                            _save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d_dropout%d_mc.pth' %
                                    (opts.member, opts.dataset, opts.output_stride, opts.random_seed, opts.dropout*100))

                        else:
                            for i in range(len(models)):
                                _save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d_cont.pth' %
                                    (opts.member[i], opts.dataset, opts.output_stride, opts.random_seed),i)
                    else:
                        _save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d_uncertainty_instance_ensembling.pth' % (opts.member[i], opts.dataset, opts.output_stride, opts.random_seed),i)

                    print("validation...")

                    for i in range(len(models)):
                        model = models[i]
                        model.eval()
                        val_score, ret_samples = validate(
                                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                                ret_samples_ids=vis_sample_id)
                        miou_history.append(val_score['Mean IoU'])
                        print(metrics.to_str(val_score))
                        if val_score['Mean IoU'] > best_score[i]:  # save best model
                            best_score[i] = val_score['Mean IoU']
                            if opts.loss_type != 'uncertainty_cross_entropy':
                                if opts.dropout != None:
                                    _save_ckpt('checkpoints/best_%s_%s_os%d_seed%d_dropout%d_mc.pth' %
                                              (opts.model, opts.dataset, opts.output_stride, opts.random_seed, opts.dropout*100))
                                else:
                                    _save_ckpt('checkpoints/best_%s_%s_os%d_seed%d_cont.pth' %
                                              (opts.member[i], opts.dataset, opts.output_stride, opts.random_seed),i)
                                    print(f'Current {opts.member[i]} best mIoU: {best_score[i]}')

                            else:
                                _save_ckpt('checkpoints/best_%s_%s_os%d_seed%d_uncertainty_instance_ensembling.pth' % (opts.member[i], opts.dataset, opts.output_stride, opts.random_seed),i)
                                print(f'Current {opts.member[i]} best mIoU: {best_score[i]}')

                    val_score,ret_samples = validate_ensemble(opts=opts,models=models,loader=val_loader,device=device,metrics=metrics)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score_ensemble:
                        best_score_ensemble = val_score['Mean IoU']

                    print(f'Current ensemble best mIoU: {best_score_ensemble}')
                    for i in range(len(models)):
                        print(f'Current {opts.member[i]} best mIoU: {best_score[i]}')

                    for model in models:
                        model.train()
                
                for scheduler in schedulers:
                    scheduler.step()

                if cur_itrs >= opts.total_itrs:
                    return
    


    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        model_seed = re.search("seed[0-9]+", opts.ckpt.split('/')[-1])[0]
        #print(model_seed)
        print(model_seed+r'_cityscapes_class_iou.json')
        val_score['Class IoU'][19] = val_score['Mean IoU']
        with open(model_seed+r'_cityscapes_class_iou.json','w') as fp:
            json.dump(val_score['Class IoU'],fp)
        return

    if opts.make_uncertainty_masks:
        model.eval()
        make_uncertainty_masks(
            opts=opts, model=model, loader=train_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        return

    interval_loss = 0
    miou_history = []
    loss_history = []
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            labels,uncertainty_masks = labels[0],labels[1]

            images = images[0].to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            #print(images.shape)
            outputs = model(images)

            if opts.loss_type == 'uce':
                #print(tuple(outputs.size()))
                y_hat_beta = torch.empty(size=[10,*list(outputs.size())])

                with torch.no_grad():
                    for i in range(10):
                        y_hat_beta[i] = model(images)

                p = torch.mean(F.softmax(y_hat_beta),dim=0)
                q = torch.std(F.softmax(y_hat_beta),dim=0)

                argmax_indices = torch.argmax(p,dim=1)

                batch_indices = torch.arange(p.size(0)).unsqueeze(1).unsqueeze(2)

                spatial_indices = torch.arange(p.size(2)).unsqueeze(0)

                sigma = q[batch_indices,argmax_indices,spatial_indices,spatial_indices]

                sigma = sigma.to(device,dtype=torch.float32)
                
                loss = criterion(sigma, outputs, labels)
                print(loss)

            elif opts.loss_type == 'uncertainty_cross_entropy':
                uncertainty_masks = torch.squeeze(uncertainty_masks)
                uncertainty_masks = uncertainty_masks.to(device,dtype=torch.float32)
                loss = criterion(outputs,labels,uncertainty_masks)

            else:
                labels = labels.squeeze(1)
                loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            loss_history.append(np_loss)
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                #if opts.loss_type != 'uncertainty_cross_entropy':
                #    if opts.dropout != None:
                #        save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d_dropout%d_mc.pth' %
                #                  (opts.model, opts.dataset, opts.output_stride, opts.random_seed, opts.dropout*100))

                #    else:
                #        save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d.pth' %
                #                  (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                #else:
                #    save_ckpt('checkpoints/latest_%s_%s_os%d_seed%d_uncertainty_instance_new.pth' % (opts.model, opts.dataset, opts.output_stride, opts.random_seed))

                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                miou_history.append(val_score['Mean IoU'])
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    if opts.loss_type != 'uncertainty_cross_entropy':
                        if opts.dropout != None:
                            save_ckpt('checkpoints/best_%s_%s_os%d_seed%d_dropout%d_mc.pth' %
                                      (opts.model, opts.dataset, opts.output_stride, opts.random_seed, opts.dropout*100))
                        else:
                            save_ckpt('checkpoints/exp_paper/best_%s_%s_os%d_seed%d.pth' %
                                      (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                            with open(f'checkpoints/exp_paper/{opts.model}_{opts.random_seed}_class_iou.json','w') as fp:
                                json.dump(val_score['Class IoU'],fp)

                    else:
                        save_ckpt('checkpoints/exp_paper/best_%s_%s_os%d_seed%d_uncertainty_instance.pth' % (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                        with open(f'checkpoints/exp_paper/{opts.model}_{opts.random_seed}_uncertainty_instance_class_iou.json','w') as fp:
                                json.dump(val_score['Class IoU'],fp)
                print(f'Current best mIoU: {best_score}')
                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                with open(f'checkpoints/exp_paper/pixel_wise/{opts.model}_{opts.random_seed}_val_miou_history.json','w') as fp:
                    json.dump(miou_history,fp)

                fig, ax = plt.subplots(1,2,figsize=(15,5))
                ax[0].plot(loss_history)
                ax[0].set_xlim(0,len(loss_history))
                ax[0].set_title('Loss History')
                ax[0].set_ylabel('Loss Value')
                ax[0].set_xlabel('Iterations')
                ax[1].plot(miou_history)
                ax[1].set_xlim(0,len(miou_history))
                ax[1].set_ylim(0,1)
                ax[1].set_title('Validation mIoU History')
                ax[1].set_ylabel('mIoU Value')
                ax[1].set_xlabel('Iterations')
                plt.tight_layout()
                if opts.loss_type != 'uncertainty_cross_entropy':
                    fig.savefig(f'{opts.model}_{opts.random_seed}_{opts.dataset}_history.png')
                else:
                    fig.savefig(f'{opts.model}_{opts.random_seed}_{opts.dataset}_uncertainty_history.png')
                return


if __name__ == '__main__':
    main()
