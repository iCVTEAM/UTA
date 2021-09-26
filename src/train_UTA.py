#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset
from net_UTA import UTA
from apex import amp


def bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    return bce.mean()


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def logMSE_loss(dpred, depth):
    mse = nn.MSELoss()
    dpred = torch.sigmoid(dpred)
    dpred = 1.0 + dpred * 255.0
    depth = 1.0 + depth * 255.0
    dpred = 257.0 - dpred
    depth = 257.0 - depth
    return mse(torch.log(dpred), torch.log(depth))


def dec_loss(pred, mask, dpred, depth):
    dpred = torch.sigmoid(dpred)
    # deeper 255 -> deeper 1
    dpred = 256.0 - dpred * 255.0
    depth = 256.0 - depth * 255.0
    # Control the error window size by kernel_size
    logDiff = torch.abs(torch.log(dpred) - torch.log(depth))
    # logDiff = torch.abs(F.avg_pool2d(torch.log(dpred) - torch.log(depth), kernel_size=7, stride=1, padding=3))
    weit = logDiff / torch.max(logDiff)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()


def train(Dataset, Network):
    # dataset
    cfg = Dataset.Config(datapath='../../data/RGBD-TR', savepath='../checkpoint/', mode='train', batch=32, lr=0.05, momen=0.9,
                         decay=5e-4, epoch=64)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=8)

    # network
    net = Network(cfg)
    net.train(True)

    # apex
    net.cuda()

    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    global_step = 0

    max_itr = cfg.epoch * len(loader)

    for epoch in range(cfg.epoch):
        if epoch < 32:
            optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr
        else:
            if epoch%2 == 0:
                optimizer.param_groups[0]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr
            else:
                optimizer.param_groups[0]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask, depth, edge) in enumerate(loader):
            # image, mask, depth = image.cuda().float(), mask.cuda().float(), depth.cuda().float()
            image, mask, depth, edge = image.float().cuda(), mask.float().cuda(), depth.float().cuda(), edge.float().cuda()
            pred1, pred2, out2h, out3h, out4h, out5h, dpred, edge2, edge3, edge4, edge5 = net(image)

            # sod loss
            loss1b = bce_loss(pred1, mask)
            loss1u = iou_loss(pred1, mask)
            loss1b_2 = bce_loss(pred2, mask)
            loss1u_2 = iou_loss(pred2, mask)
            loss2s_b = bce_loss(out2h, mask)
            loss2s_u = iou_loss(out2h, mask)
            loss3s_b = bce_loss(out3h, mask)
            loss3s_u = iou_loss(out3h, mask)
            loss4s_b = bce_loss(out4h, mask)
            loss4s_u = iou_loss(out4h, mask)
            loss5s_b = bce_loss(out5h, mask)
            loss5s_u = iou_loss(out5h, mask)

            # depth error-weighted correction loss
            loss1h = dec_loss(pred1, mask, dpred, depth)
            loss1h_2 = dec_loss(pred2, mask, dpred, depth)
            loss2h = dec_loss(out2h, mask, dpred, depth)
            loss3h = dec_loss(out3h, mask, dpred, depth)
            loss4h = dec_loss(out4h, mask, dpred, depth)
            loss5h = dec_loss(out5h, mask, dpred, depth)

            # depth loss
            loss2d = logMSE_loss(dpred, depth)

            # edge loss
            loss2e = bce_loss(edge2, edge)
            loss3e = bce_loss(edge3, edge)
            loss4e = bce_loss(edge4, edge)
            loss5e = bce_loss(edge5, edge)

            loss = loss2d + loss1b + loss1u + loss1h + loss2e + loss3e + loss4e + loss5e + loss1b_2 + loss1u_2 + loss1h_2\
                   + 0.8 * (loss2s_b + loss2s_u + loss2h) \
                   + 0.6 * (loss3s_b + loss3s_u + loss3h) \
                   + 0.4 * (loss4s_b + loss4s_u + loss4h) \
                   + 0.2 * (loss5s_b + loss5s_u + loss5h)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
               scale_loss.backward()
            # loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            if step % 60 == 0:
                print('%s | step:%d/%d | lr=%.6f | loss=%.3f | s=%.3f | u=%.3f | d=%.3f | h=%.3f | e=%.3f ' % (
                    datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[1]['lr'],
                    loss.item(), loss1b.item(), loss1u.item(), loss2d.item(), loss1h.item(), loss2e.item()))

        if epoch >= 31:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


if __name__ == '__main__':
    train(dataset, UTA)
