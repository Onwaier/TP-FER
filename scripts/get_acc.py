from torchvision import transforms
import torch
import os
import sys
import cv2
import numpy as np
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from util.raf_dataset_util import RAFDataSet
from util.ferplus_dataset_util import FERplusDataSet, FER2013DataSet
from util.affectnet_dataset_util import AffectNetDataset
from util.common_util import acc_for_one_label
from model.resnet import resnet18

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RAF', help='dataset.')
    parser.add_argument('--checkpoint_path', type=str, default='', help='The checkpoint file is located in the checkpoints directory')
    return parser.parse_args()
    
def get_new_acc():
    args = parse_args()
    dataset_name = args.dataset
    checkpoint_path = os.path.join('../checkpoints', args.checkpoint_path)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    batch_size = 64
    workers = 4
    dataset_path = {
        'RAF':'/data/ljj/FER/RAF',
        'FER2013':'/data/ljj/FER/FERPlus',
        'Affect':'/data/ljj/FER/AffectNet'
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    num_classes = 7
    if dataset_name == 'RAF':
        val_dataset = RAFDataSet(dataset_path[dataset_name], phase = 'test', transform = data_transforms_val)    
    elif dataset_name == 'FER2013':
        val_dataset = FER2013DataSet(dataset_path[dataset_name], phase = 'PrivateTest', transform = data_transforms_val) 
        num_classes = 7
    elif dataset_name == 'Affect':
        val_dataset = AffectNetDataset(dataset_path[dataset_name], phase = 'test', transform = data_transforms_val, basic_aug = False, with_align=True, num_classes = 8)
        num_classes = 8
    print('Validation set size:', val_dataset.__len__())
    
    if dataset_name == 'Affect':
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               pin_memory = True)
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size = batch_size,
                                                num_workers = workers,
                                                shuffle = False,  
                                                pin_memory = True)
    model = resnet18(pretrained = False, num_classes = num_classes)
    # load weights
    pretrained = torch.load(checkpoint_path)
    pretrained_state_dict = pretrained['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict, strict = False)
    model = model.cuda()
    res_acc = 0.0
    sample_num = 0
    with torch.no_grad():
        model.eval()
        for batch_i, (imgs, targets, indexes, targets2) in enumerate(val_loader):
            outputs = model(imgs.cuda())
            targets = targets.cuda()
            tmp_acc = acc_for_one_label(outputs, targets)
            print('tmp_acc', tmp_acc)
            sample_num += imgs.shape[0]
            res_acc += tmp_acc[0].item() * imgs.shape[0]
    print('res_acc', res_acc / sample_num)
    print('sample_num', sample_num)


if __name__ == '__main__':
    get_new_acc()
