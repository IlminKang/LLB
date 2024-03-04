import os
import sys
import argparse
import time
import torch
import torch.nn as nn

from utils import general, multiprocess, losses
import dataset
import models
from core import engine

def main(args):

    # settings
    device = torch.device(args.device)

    multiprocess.init_distributed_mode(args)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    print('>>>Using Distributed Data Parallel...')

    if args.seed:
        current_seed = general.fix_seed(args.seed)
    else:
        current_seed = 'RandomSeed'

    args.class_num ,args.data_dir ,args.img_size = general.dataset_qualification(args)
    args.vit_name = general.model_qualification(args)

    if args.local_rank == 0:
        model_name = general.get_model_name(args, current_seed)
    else:
        pass

    # weights and bias
    if args.wandb and args.local_rank == 0:
        pass
    else:
        pass

    print('Done!\n')

    # Load dataset
    print(f'>>>Using {args.dataset}, loading data from "{args.data_dir}"...')
    print(f'\t|-Image size: {args.img_size}')

    train_set = dataset.get_dataset(args, split='train')
    val_set = dataset.get_dataset(args, split='val')
    test_set = dataset.get_dataset(args, split='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=250,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=True)
    if val_set:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler,
                                                 drop_last=True)
    print('Done!\n')

    # loading network
    print(">>>Creating network...")
    model = models.get_model(args, device)
    model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # , find_unused_parameters=True
    model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = general.get_optimizer(args, params)
    lr_scheduler = general.get_scheduler(args, optimizer)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        print(f'continue training from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    print('Done!\n')


    if not args.eval:
        print(">>>Start training...")
        best_acc = 0
        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                train_sampler.set_epoch(epoch)
                test_sampler.set_epoch(epoch)

                print(f'Epoch: [{epoch}] LR{optimizer.param_groups[0]["lr"]:.10f}')

                train_loss, train_top1, train_ce_loss, train_ent_loss = engine.train_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    device=device,
                    epoch=epoch,
                    print_freq=500,
                    scaler=scaler, )

                test_loss, test_top1, test_ce_loss, test_ent_loss = engine.eval_epoch(
                    model=model,
                    data_loader=test_loader,
                    device=device,
                    epoch=epoch,
                    print_freq=25,)

                lr_scheduler.step()

                if test_top1 > best_acc:
                    best_acc = test_top1
                    if args.local_rank == 0 and args.save:
                        general.save_checkpoint(state={'epoch': epoch + 1,
                                                       'args': args,
                                                       'model': model_without_ddp.state_dict(),
                                                       'optimizer': optimizer.state_dict(),
                                                       'lr_scheduler': lr_scheduler.state_dict(),
                                                       'scaler': scaler.state_dict()},
                                                is_best=True,
                                                save_root=args.save_dir,
                                                filename='model_best.pth')

    else:
        print(">>>Start evaluating...")
        test_loss, test_top1 = engine.eval_epoch(model=model,
                                                 data_loader=test_loader,
                                                 device=device,
                                                 epoch=0,
                                                 print_freq=25)

        if args.local_rank == 0 and args.save:
            general.save_checkpoint(state={'epoch': 0,
                                           'model': model_without_ddp.state_dict(),},
                                    is_best=True, save_root=args.save_dir, filename='model_eval.pth')

        print(f'{args.method} evaluation top1: {test_top1:3.4f} \t loss: {test_loss:3.6f}')


if __name__ == '__main__':
    args = general.get_args_parser().parse_args()
    main(args)