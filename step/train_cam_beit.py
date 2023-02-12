
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
import timm
import numpy as np

import voc12.dataloader
from misc import pyutils, torchutils


def freeze_timm_models(model):
    train_list = ['fc_norm', 'head']
    for name, param in model.named_parameters():
        if not sum([item in name for item in train_list]):
            if param.requires_grad:
                param.requires_grad = False
    return model

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda(non_blocking=True)
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):

    # model = getattr(importlib.import_module(args.cam_network), 'Net')()

    NUM_CLASSES = 20
    model = timm.create_model('beitv2_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    model = freeze_timm_models(model)
    
    # train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
    #                                                             resize_long=(320, 640), hor_flip=True,
    #                                                             crop_size=512, crop_method="random")
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(224, 448), hor_flip=True,
                                                                crop_size=224, crop_method="random")
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = int(np.ceil(len(train_dataset) / args.cam_batch_size) * args.cam_num_epoches)

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=224)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # param_groups = model.trainable_parameters()
    # optimizer = torchutils.PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    #     {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    # ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    
    optimizer = torchutils.AdamWOptimizer([
        {'params': model.fc_norm.parameters(), 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': model.head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_step+1, eta_min=0, last_epoch=-1, verbose=False)
    
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            
            # print(img.shape)
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (np.mean(scheduler.get_last_lr())),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '_beit.pth')
    torch.cuda.empty_cache()
