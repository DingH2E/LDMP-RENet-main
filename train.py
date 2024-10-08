import argparse
import logging
import os
import random
import time

import numpy as np

import torch.cuda
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter


from model.LERENet import FSSNet
from util import config, transform, dataset
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, backbone_optimizer

def get_parser():
    parser = argparse.ArgumentParser(description='LERENet')
    parser.add_argument('--config', type=str, default='config/fold2_resnet50.yaml', help='config file')

    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg


def get_logger():
    logger_name = "master_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format_ = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(format_))
    logger.addHandler(handler)
    return logger


def main_process():
    return True


def train_gpu():
    args = get_parser()
    assert args.classes > 1

    """Set seeds"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.deterministic = True
        # cudnn.benchmark = True   # it will accelerate if network is fixed but will expand giant time
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    """Start"""
    worker(args)


def worker(args_):
    global args
    args = args_
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    """Initial model and optimizer"""
    model = Test1Net(layers=args.layers, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                   pretrained=args.pretrained, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg,
                     align=args.align)
    optimizer = backbone_optimizer(model, args)

    """Logger and Writer"""
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("-> creating model...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    """Parallel computing"""
    model = model.cuda()

    """Load weight to finetune or test"""
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("-> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("-> loaded weight '{}'".format(args.weight))
        else:
            logger.info("-> no weight found at '{}'".format(args.weight))

    """Load checkpoint to train continuously"""
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("-> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['state_dict'])
            logger.info("-> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("-> no check point found at '{}'".format(args.resume))

    """Normalization"""
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    assert args.split in [0, 1, 2, 999]

    """Set transform, train data and trainLoader for training"""
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)

    train_data = dataset.SemData(split=args.split, shot=args.shot,
                                 normal=args.normal, data_root=args.data_root,
                                 data_list=args.train_list, nom_list=args.trainnom_list,
                                 transform=train_transform, mode='train')
    # train_data = dataset2.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
    #                               data_list=args.train_list,
    #                               transform=train_transform, mode='train')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=train_sampler, drop_last=True)

    """Set transform, train data and trainLoader for evaluating"""
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split=args.split, shot=args.shot,
                                   normal=args.normal, data_root=args.data_root,
                                   data_list=args.val_list, nom_list=args.valnom_list,
                                   transform=val_transform, mode='val')
        # val_data = dataset2.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
        #                             data_list=args.val_list,
        #                             transform=val_transform, mode='val')

        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    """Target aim"""
    max_iou = 0
    max_fbiou = 0
    best_epoch = 0
    filename = 'TLDNet.pth'

    """Schedule"""
    for epoch in range(args.start_epoch, args.epochs):
        # Set seed
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        # train
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
        # test
        if args.evaluate and (epoch % 2 == 0 or (args.epochs <= 50 and epoch % 1 == 0)):
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                best_epoch = epoch
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if mIoU_val > max_fbiou :
                max_fbiou = mIoU_val

            logger.info('Best Epoch {:.1f} Best IoU {:.4f} Best FB-IoU {:.4f}'.format( best_epoch, max_iou, max_fbiou))

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


def train(train_loader, model, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    print('Warmup: {}'.format(args.warmup))

    for i, (input, target, nomimg, s_input, s_mask, seeds, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)  # [b,1,200,200]
        s_mask = s_mask.cuda(non_blocking=True)    # [b,1,200,200]
        input = input.cuda(non_blocking=True)      # [b,3,200,200]
        target = target.cuda(non_blocking=True)    # [b,200,200]
        # nomimg = nomimg.cuda(non_blocking=True)
        # predicted mask[b,200,200] loss [1,b]
        # output, main_loss = model(s_x=s_input, s_y=s_mask, nom=None, x=input, y=target)
        output, main_loss = model(q=input, s=s_input, s_mask=s_mask, q_mask=target)
        """Alternative"""
        loss = main_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """Computing I and U"""
        n = input.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        """Computing acc and so on"""
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        loss_meter.update(loss.item(), n)
        # aux_loss_meter.update(aux_loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        """Print information on time"""
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '                        
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = 4
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap
    # Set test seeds
    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    # Set numbers
    test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, nomimg, s_input, s_mask, seeds, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)  # [1,3,200,200]
            target = target.cuda(non_blocking=True)  # [1,200,200]
            ori_label = ori_label.cuda(non_blocking=True)  # original image dim
            start_time = time.time()
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            # nomimg = nomimg.cuda(non_blocking=True)
            # output = model(s_x=s_input, s_y=s_mask, nom=None, x=input, y=target)
            output = model(q=input, s=s_input, s_mask=s_mask, q_mask=target)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            # utilize initial images and corresponding mask to test
            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            loss = criterion(output, target)

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
            class_union_meter[(subcls - 1) % split_gap] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num / 100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
    # Computing target aim
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__' :
    train_gpu()
