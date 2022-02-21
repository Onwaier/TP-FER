import math
import numpy as np
import os
import torchvision.models as models
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os
import sys
import shutil
import torch
import datetime
import time
import torch.nn as nn
from util import image_utils
import argparse
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model.resnet import resnet18
from util.plot_util import plot_confusion_matrix
from util.common_util import initialize_weight_goog
from util.raf_dataset_util import RAFDataSet
from util.ferplus_dataset_util import FERplusDataSet, FER2013DataSet
from util.affectnet_dataset_util import AffectNetDataset
from util.sampler import ImbalancedDatasetSampler
from util.sfew_dataset_util import SFEWDataSet
from util.common_util import AverageMeter, ProgressMeter, RecorderMeter, my_cross_entropy, setup_seed, write_label_log, find_no_clean_samples
from util.loss_util import FocalLoss, CELoss
from util.randaugument import RandomAugment
from util import transform as T

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-[%S]-")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RAF', help='dataset.')
    parser.add_argument('--raf_path', type=str, default='/data/ljj/FER/RAF', help='Raf-DB dataset path.')
    parser.add_argument('--ferplus_path', type=str, default='/data/ljj/FER/FERPlus', help='FERPlus dataset path.')
    parser.add_argument('--affectnet_path', type=str, default='/data/ljj/FER/AffectNet/', help='AffectNet dataset path.')
    parser.add_argument('--sfew_path', type=str, default='/data/ljj/FER/SFEW2.0/', help='SFEW2.0 dataset path.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default='/data/ljj/project/RAN/checkpoint/ijba_res18_naive.pth.tar',
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--margin_1', type=float, default=0.15, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--scheduler', type=str, default="exp", help='Scheduler, expLR or stepLR.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--cm_path', type=str, default='./log/' + time_str + 'cm.png')        
    parser.add_argument('--best_cm_path', type=str, default='./log/' + time_str + 'best_cm.png')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/' + time_str + 'model.pth')
    parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoints/'+time_str+'model_best.pth')
    parser.add_argument('--focal_loss', action="store_true", \
        help='if true using focal loss, or not using, default is false')
    parser.add_argument('--ce_loss', action="store_true", \
        help='if true using CE loss, or not using, default is false')
    parser.add_argument('--record_relabel', action="store_true", \
        help='if true, record the relabel process, or not recording, default is false')
    parser.add_argument('--setup_seed', default=7777, type=int, help='setup seed')
    parser.add_argument('--alpha1', type=float, default=0.6, help='the min score of facial expression')
    parser.add_argument('--alpha2', type=float, default=0.8, help='the min score of facial expression') 
    parser.add_argument('--warm_epoch', default=5, type=int, help='warm epoch')
    parser.add_argument('--relabel_epoch', type=int, default=10, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--in_turn', action="store_true", \
        help='if true using the strategy of in-turn, or not using, default is false')
    parser.add_argument('--relabel_every_epoch', default=1, type=int, help='relabel at every epoch')
    parser.add_argument('--step_size', default=10, type=int, help='the setting of step size in sgd')
    parser.add_argument('--gamma', default=0.9, type=float, help='the setting of gamma in adam')
    parser.add_argument('--with_align', action="store_true", \
        help='if true, train with align images, or not using, default is true')
    parser.add_argument('--num_classes', default=8, type=int, help='the num of class')
    parser.add_argument('--transform_type', default=1, type=int, help='if value is 0, choose data_transforms; value is 1, choose fer2013data_transforms') 
    parser.add_argument('--no_weight', action="store_true", \
        help='if true, resnet without weight module, or using weight module, default is false') 
    parser.add_argument('--noise_file', type=str, default='', help='train with noise file')
    parser.add_argument('--desc', type=str, default='', help='description about command')
    parser.add_argument('--label_log', action="store_true", \
        help='if true, write label log, or not writing, default is false')
    parser.add_argument('--no_two_label_loss', action="store_true", \
        help='if true, not using two label loss, or using, default is false')    
    return parser.parse_args()
    
     
args = parse_args()
command_str = 'python ' + ' '.join(sys.argv)
args.desc = command_str
print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')
best_acc = 0
class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral', 'Contempt']
alpha = [0.895, 0.977, 0.942, 0.611, 0.838, 0.943, 0.794]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def run_training():
    global best_acc
    imagenet_pretrained = False
    need_weight = not args.no_weight
    res18 = resnet18(pretrained = imagenet_pretrained, drop_rate = args.drop_rate, num_classes = args.num_classes, need_weight = need_weight) 
    setup_seed(args.setup_seed)
    
    # init weight
    if not imagenet_pretrained:
         for m in res18.modules():
            initialize_weight_goog(m)

    # load weights of the pretrained model       
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                new_key = key[7:]
                model_state_dict[new_key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False)  
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            # best_acc = checkpoint['best_acc']
            res18.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # data_transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    fer2013data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            T.PadandRandomCrop(border=4, cropsize=(224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    transforms_list = [data_transforms, fer2013data_transforms]
    if args.dataset == 'RAF':
        train_dataset = RAFDataSet(args.raf_path, phase = 'train', transform = transforms_list[args.transform_type], basic_aug = True, noise_file=args.noise_file)    
    elif args.dataset == 'FERPlus':
        train_dataset = FER2013DataSet(args.ferplus_path, phase = 'Training', transform = transforms_list[args.transform_type])     
    elif args.dataset == 'Affect':
        train_dataset = AffectNetDataset(args.affectnet_path, phase = 'train', transform = transforms_list[args.transform_type], basic_aug = True, with_align=args.with_align, num_classes = args.num_classes)
    elif args.dataset == 'SFEW':
        train_dataset = SFEWDataSet(args.sfew_path, phase = 'Training', transform = transforms_list[args.transform_type], basic_aug = True)
    else:
        raise ValueError("dataset not supported.")
        
    print('Train set size:', train_dataset.__len__())

    if args.dataset == 'Affect':
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                sampler = ImbalancedDatasetSampler(train_dataset),
                                                batch_size = args.batch_size,
                                                num_workers = args.workers, 
                                                pin_memory = True)
    elif args.dataset == 'RAF' or args.dataset == 'FERPlus' or args.dataset == 'SFEW':
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
    else:
        raise ValueError("dataset not supported.")

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    if args.dataset == 'RAF':                                         
        val_dataset = RAFDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)    
    elif args.dataset == 'FERPlus':
        val_dataset = FER2013DataSet(args.ferplus_path, phase = 'PrivateTest', transform = data_transforms_val) 
    elif args.dataset == 'Affect':
        val_dataset = AffectNetDataset(args.affectnet_path, phase = 'test', transform = data_transforms_val, basic_aug = False, with_align=args.with_align, num_classes = args.num_classes)
    elif args.dataset == 'SFEW':
        val_dataset = SFEWDataSet(args.sfew_path, phase = 'Val', transform = data_transforms_val)
    else:
        raise ValueError("dataset not supported.") 

    print('Validation set size:', val_dataset.__len__())

    if args.dataset == 'Affect':
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                # sampler = ImbalancedDatasetSampler(val_dataset),
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               pin_memory = True)
    else:    
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size = args.batch_size,
                                                num_workers = args.workers,
                                                shuffle = False,  
                                                pin_memory = True)
        
    params = res18.parameters()
    filter_list = ['fc.weight', 'fc.bias']
    base_parameters_model = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, res18.named_parameters()))))


    if args.optimizer == 'adam':
        if args.dataset == 'RAF':
            optimizer = torch.optim.Adam([{'params': base_parameters_model}, {'params': list(res18.fc.parameters()), 'lr': args.lr}], weight_decay = 1e-4, lr=1e-3)
        else:
            optimizer = torch.optim.Adam(params,weight_decay = 1e-4, lr=args.lr)
        
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 5e-4)
    else:
        raise ValueError("Optimizer not supported.")
        
    if args.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError("Scheduler not supported.") 

    recorder = RecorderMeter(args.epochs)
    res18 = res18.cuda()
    if args.focal_loss:
        criterion = FocalLoss(7, gamma = 0.75)
    elif args.ce_loss:
        criterion = CELoss(7, alpha = alpha, use_alpha = True)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss(reduce = True, size_average = True) 
    # margin_1 = args.margin_1
    # margin_2 = args.margin_2
    # beta = args.beta

    if args.label_log:
        label_log_name = './log/' + time_str + 'label_log.txt'
        write_label_log(train_loader.dataset, label_log_name)

    # 记录训练时的参数值
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('--------args----------\n')
        for k in list(vars(args).keys()):
            f.write('%s: %s\n' % (k, vars(args)[k]))
        f.write('--------args----------\n')  

    for i in range(0, args.epochs):
        # adjust_learning_rate(optimizer, i)
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        print('log: ', time_str + 'log.txt')
        print('command: ', args.desc)

        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n') 
        train_acc, train_los = train(train_loader, res18, criterion, criterion2, optimizer, scheduler, i)
        val_acc, val_los = validate(val_loader, res18, criterion, criterion2, optimizer, i)

        recorder.update(i, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        print('Current best accuracy: ', best_acc)
        end_time = time.time()
        print('run time of the epoch:', end_time - start_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc) + '\n')
            f.write('run time of the epoch: %s s\n'%(end_time - start_time))

    if args.label_log:
        label_log_name = './log/' + time_str + 'label_log.txt'
        write_label_log(train_loader.dataset, label_log_name)
        
# 训练阶段     
def train(train_loader, model, criterion, criterion2, optimizer, scheduler, epoch):
    running_loss = 0.0
    correct_sum = 0
    iter_cnt = 0
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             time_str,
                             prefix="Epoch: [{}]".format(epoch))
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta

    model.train()

    for batch_i, (imgs, targets, indexes, targets2) in enumerate(train_loader):
        batch_sz = imgs.size(0) 
        iter_cnt += 1
        tops = int(batch_sz * beta)
        optimizer.zero_grad()
        imgs = imgs.cuda()
        outputs = model(imgs)
        
        targets = targets.cuda()
        targets2 = targets2.cuda()
        img_num = 0

        if epoch < args.warm_epoch:
            if args.no_two_label_loss:
                loss = criterion(outputs, targets)
            else:
                loss = 0.5 * criterion(outputs, targets) + 0.5 * criterion(outputs, targets2)
            img_num = targets.shape[0]
        elif epoch >= args.warm_epoch:
            if args.in_turn and epoch % 2 == 1:
                if args.no_two_label_loss:
                    loss = criterion(outputs, targets)
                else:
                    loss = 0.5 * criterion(outputs, targets) + 0.5 * criterion(outputs, targets2)
                img_num = targets.shape[0]
            else:
                sm = torch.softmax(outputs, dim = 1)
                values, indices = sm.topk(2, dim = 1, largest = True, sorted = True)
                idx1 = torch.nonzero((values[:, 0] - values[:, 1] > margin_2) & (values[:, 0] >= args.alpha1)).squeeze(1)
                idx2 = torch.nonzero((values[:, 0] - values[:, 1] <= margin_2) & (values[:, 0] + values[:, 1] >= args.alpha2)).squeeze(1)
                idx = torch.cat([idx1, idx2], dim = 0)

                # if epoch == args.epochs - 1:
                    # no_clean_samples_log_name = './log/' + time_str + 'no_clean_samples.txt'
                    # find_no_clean_samples(idx, indexes, train_loader.dataset, no_clean_samples_log_name)

                if idx.shape[0] == 0:
                    continue
                else:
                    if args.no_two_label_loss:
                        loss = criterion(outputs[idx], targets[idx])
                    else:
                        loss = 0.5 * criterion(outputs[idx], targets[idx]) + 0.5 * criterion(outputs[idx], targets2[idx])

                img_num = idx.shape[0]

        if epoch >= args.relabel_epoch:
            if (epoch - args.relabel_epoch) % args.relabel_every_epoch == 0:
                sm = torch.softmax(outputs, dim = 1)
                values, indices = sm.topk(2, dim = 1, largest = True, sorted = True)
                idx1 = torch.nonzero((values[:, 0] - values[:, 1] > margin_2) & (values[:, 0] >= args.alpha1)).squeeze(1)
                idx2 = torch.nonzero((values[:, 0] - values[:, 1] <= margin_2) & (values[:, 0] + values[:, 1] >= args.alpha2)).squeeze(1)
                update_idx1 = indexes[idx1]
                update_idx2 = indexes[idx2]
                if args.dataset == 'Affect':
                    train_loader.dataset.tmp_label[update_idx1.cpu().numpy()] = indices[idx1, 0].cpu().numpy()
                    train_loader.dataset.tmp_label2[update_idx1.cpu().numpy()] = indices[idx1, 0].cpu().numpy()
                    train_loader.dataset.tmp_label[update_idx2.cpu().numpy()] = indices[idx2, 0].cpu().numpy()
                    train_loader.dataset.tmp_label2[update_idx2.cpu().numpy()] = indices[idx2, 1].cpu().numpy() 
                else:
                    train_loader.dataset.label[update_idx1.cpu().numpy()] = indices[idx1, 0].cpu().numpy()
                    train_loader.dataset.label2[update_idx1.cpu().numpy()] = indices[idx1, 0].cpu().numpy()
                    train_loader.dataset.label[update_idx2.cpu().numpy()] = indices[idx2, 0].cpu().numpy()
                    train_loader.dataset.label2[update_idx2.cpu().numpy()] = indices[idx2, 1].cpu().numpy()  
        else:
            pass

        loss.backward()
        optimizer.step()
        
        running_loss += loss
        _, predicts = torch.max(outputs, 1)
        correct_num = torch.eq(predicts, targets).sum()
        correct_sum += correct_num

        prec1 = accuracy(outputs.data, targets, topk=(1,))
        losses.update(loss.item(), img_num)
        top1.update(prec1[0].item(), imgs.size(0))

        if batch_i % args.print_freq == 0:
            progress.display(batch_i)

    if args.dataset == 'Affect':
        train_loader.dataset.label = train_loader.dataset.tmp_label.copy()
        train_loader.dataset.label2 = train_loader.dataset.tmp_label2.copy() 

    scheduler.step()
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    running_loss = running_loss/iter_cnt
    print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (epoch, acc, running_loss))
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (epoch, acc, running_loss))
    if args.dataset == 'Affect':
        state = {'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'train_acc': acc,
                            'optimizer': optimizer.state_dict()}
        checkpoint_path = './checkpoints/'+ time_str + str(epoch) + '_' + str(acc.item()) + '_model.pth'
        torch.save(state, checkpoint_path) 
    return top1.avg, losses.avg

# 测试阶段
def validate(val_loader, model, criterion, criterion2, optimizer, epoch):
    global best_acc
    with torch.no_grad():
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Accuracy', ':6.3f')
        progress = ProgressMeter(len(val_loader),
                                [losses, top1],
                                time_str,
                                prefix='Test: ')
        if args.num_classes == 7:
            class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
        elif args.num_classes == 8:
            class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral', 'Contempt']
            if args.dataset == 'Affect':
                class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'] # for AffectNet

        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        model.eval()
        for batch_i, (imgs, targets, _, new_targets) in enumerate(val_loader):
            outputs = model(imgs.cuda())
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(outputs, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += outputs.size(0)

            prec1 = accuracy(outputs.data, targets, topk=(1,))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1[0].item(), imgs.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            if batch_i == 0:
                all_predicted = predicted
                all_targets = targets
            else:
                all_predicted = torch.cat((all_predicted, predicted),0)
                all_targets = torch.cat((all_targets, targets),0)
            
            if batch_i % args.print_freq == 0:
                progress.display(batch_i)

        running_loss = running_loss/iter_cnt   
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f: 
           f.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f\n" % (epoch, acc, running_loss)) 

        is_best = top1.avg > best_acc
        if is_best:
            best_acc = top1.avg


        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict()},
                        is_best, args)

        # Compute confusion matrix
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                            title= ' Confusion Matrix (Accuracy: %.3f%%)'%(top1.avg))
        plt.savefig(args.cm_path)
        if is_best:
            shutil.copyfile(args.cm_path, args.best_cm_path)
        plt.close()

        return top1.avg, losses.avg

# 获取精度
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        # maxk = max(topk) 
        # batch_size = target.size(0)
        # pred_prob, pred_val = output.topk(maxk, 1, True, True)
        # idx1 = torch.nonzero(pred_prob[:, 0] - pred_prob[:, 1] > 0.2).squeeze(1)
        # idx2 = torch.nonzero(pred_prob[:, 0] - pred_prob[:, 1] <= 0.2).squeeze(1)
        # cnt = 0.0
        # res = []
        # if idx1.dim() == 1 and idx1.shape[0] != 0:
        #     correct = pred_val[idx1, 0].eq(target[idx1].expand_as(pred_val[idx1, 0]))
        #     cnt += correct.reshape(-1).float().sum(0, keepdim=True)
        # if idx2.dim() == 1 and idx2.shape[0] != 0:
        #     correct = pred_val[idx2, :].eq(target[idx2].view(-1, 1).expand_as(pred_val[idx2, :]))
        #     cnt += correct.reshape(-1).float().sum(0, keepdim=True)
        # res.append(cnt.mul_(100 / batch_size))
        
        return res

# 保存模型
def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

if __name__ == "__main__":                    
    run_training()
