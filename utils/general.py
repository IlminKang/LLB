import os
import torch
import random
import numpy as np
import shutil
import argparse
import torch.nn as nn


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Project Name", add_help=add_help)


    # Training parameters
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='usable device number')
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting point of epochs')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='objective function')
    parser.add_argument("--opt", type=str, default='Adam', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr_scheduler", default="steplr", type=str, help="lr scheduler (default: steplr)")
    parser.add_argument("--lr_step", default=100, type=int,help="decrease lr every step-size epochs (steplr scheduler only)")
    parser.add_argument("--lr_steps", default=[10, 30, 50], nargs="+", type=int,help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr_gamma", default=0.1, type=float,help="decrease lr by a factor of lr-gamma")
    parser.add_argument('--warmup', action='store_true', help='lr scheduler warmup')
    parser.add_argument("--t_max", default=10, type=int, help="Maximum number of iterations.")
    parser.add_argument("--eta_min", default=0.00005, type=float, help="Minimum learning rate")
    parser.add_argument("--last_epoch", default=-1, type=int, help="The index of last epoch")
    parser.add_argument('--save_dir', type=str, default='', help='output save dir')
    parser.add_argument('--save', action='store_true', help='save models')
    parser.add_argument("--resume", default=False, type=bool, help="restart training")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--wandb", action='store_true', help="use wandb")
    parser.add_argument("--eval", action='store_true', help="only evaluation")

    # Method
    parser.add_argument('--method', type=str, default='NViT', help='Training ViT backbone method')

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Encoder parameters
    parser.add_argument('--encoder', type=str, default='ViT', help='base encoder type')
    parser.add_argument('--transfer', action='store_true', help='use pre-trained weight')
    parser.add_argument('--encoder_weight', type=str, default='', help='path to encoder weight')
    parser.add_argument('--rep', action='store_true', help='use reproduced pre-trained weight')

    # Distributed training parameters
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='type of dataset')
    parser.add_argument('--data_dir', type=str, default=None, help='path to dataset directory')
    parser.add_argument('--class_num', type=int, default=None, help='number of class labels')
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument('--num_workers', type=int, default=4, help='size of workers')
    parser.add_argument('--norm', action='store_true', help='normalize image')

    # Vit parameters
    parser.add_argument('--vit_size', type=str, default='base', help='Vit model size')
    parser.add_argument('--vit_patch', type=int, default=16, help='Vit patch size')
    parser.add_argument('--vit_resolution', type=int, default=224, help='Vit resolution size')
    parser.add_argument('--vit_ft', type=bool, default=False, help='Vit fine-tune (in21k)')

    # NViT parameters object_size
    parser.add_argument('--object_size', type=int, default=512, help='Num non-visual embeddings')
    parser.add_argument('--num_nvit_layers', type=int, default=6, help='Num nvit layers')
    parser.add_argument('--target_layer', type=int, default=11, help='Num nvit layers')
    parser.add_argument('--initialize', type=str, default='one_hot', help='one_hot, xavier')
    parser.add_argument('--train_emb', action='store_true', help='trainable non-visual features')
    parser.add_argument('--nv_weights', action='store_true', default=False, help='pre-trained LLB weights')
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.1)



    return parser



def dataset_qualification(args):
    '''
    :param args: dataset
    :return: number of class, location of dataset, image size
    '''
    if args.dataset.lower() == 'imagenet':
        class_num = 1000
        data_dir = '' # path to ImageNet1K
        img_size = 224
    elif args.dataset.lower() == 'places365':
        class_num = 365
        data_dir = '' # path to Places365
        img_size = 384
    elif args.dataset.lower() == 'inaturalist2018':
        class_num = 8142
        data_dir = '' # path to iNaturalist2018
        img_size = 224
    else:
        raise TypeError(f'Invalid dataset! : {args.dataset}')

    if data_dir == '':
        raise ValueError(f'Invalid dataset directory! Got empty path! Please finish configuration...')

    return class_num, data_dir, img_size


def model_qualification(args):
    '''
    :param args: model hyper-parameters
    :return:
    '''
    if args.encoder.lower() == 'vit':
        sizes = ['base', 'large', 'tiny']
        if args.vit_size not in sizes:
            raise TypeError(f'Invalid encoder ViT size: expected {sizes} but got {args.vit_size}...')

        vit_name = f'vit-{args.vit_size}-patch{args.vit_patch}-{args.vit_resolution}'
        if args.vit_ft:
            vit_name += f'-in21k'
    else:
        raise TypeError(f'Invalid encoder type!: {args.encoder}')

    return vit_name


def get_optimizer(args, params):
    '''
    :param args: optimizer hyper-parameters
    :param params: model parameters
    :return: optimizer
    '''
    print(f'\t|-Optimizer: {args.opt}')
    if args.opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f'\t\t|-lr:{args.lr}, momentum:{args.momentum}, weight_decay:{args.weight_decay}')
    elif args.opt.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        print(f'\t\t|-lr:{args.lr}, weight_decay:{args.weight_decay}')
    elif args.opt.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        print(f'\t\t|-lr:{args.lr}, weight_decay:{args.weight_decay}')
    else:
        raise TypeError(f'Invalid optimizer type! : {args.opt.lower()}')

    return optimizer


def get_scheduler(args, optimizer):
    '''
    :param args: scheduler hyper-parameter
    :return: scheduler
    '''
    print(f'\t|-Scheduler: {args.lr_scheduler}')
    if args.lr_scheduler.lower() == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        print(f'\t\t|-lr_step:{args.lr_step}, lr_gamma:{args.lr_gamma}')
    elif args.lr_scheduler.lower() == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        print(f'\t\t|-lr_steps:{args.lr_steps}, lr_gamma:{args.lr_gamma}')
    elif args.lr_scheduler.lower() == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.t_max, eta_min=args.eta_min)
        print(f'\t\t|-CosineAnnealingLR: t_max:{args.t_max}, eta_min:{args.eta_min}')
    elif args.lr_scheduler.lower() == 'exponentiallr':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise TypeError(f'Invalid Scheduler type! : {args.lr_scheduler.lower()}')

    if args.warmup:
        from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

        lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                    warmup_start_value=5e-3,
                                                    warmup_end_value=args.lr,
                                                    warmup_duration=2)

    return lr_scheduler

def fix_seed(seed):
    '''
    :param seed: seed
    :return: current seed string type
    '''
    print(f'>>>Fixing seed : {seed}')
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    return str(random_seed)


def get_model_name(args, current_seed):
    '''
    :param args: hyper-parameters
    :param current_seed: current seed state
    :return: model name
    '''
    
    names = [args.method,
             args.dataset,
             args.vit_size,
             f'L{args.num_nvit_layers}',
             f'Fr{args.target_layer}',
             f'Obj{args.object_size}',]
    model_name = '_'.join(names)

    print(f'>>>Training {model_name}...')

    make_save_dir(args, model_name)
    return model_name



def make_save_dir(args, model_name):
    '''
    :param args: hyper-parameters
    :param model_name: model name
    :return:
    '''
    dir_path = f'{args.save_dir}/{model_name}'
    if not os.path.isdir(f'{args.save_dir}'):
        os.mkdir(f'{args.save_dir}')

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    print(f'\t|-Saving model to {dir_path}')
    args.save_dir = dir_path


def save_checkpoint(state, is_best, save_root, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print(f'Saving checkpoint to {save_root}/{filename}...')
        torch.save(state, f'{save_root}/{filename}')
