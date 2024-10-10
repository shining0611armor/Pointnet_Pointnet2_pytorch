"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datasets.tta_datasets as tta_datasets

import datetime
import logging
import provider
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--dataset_name', default='modelnet', help='model name [default: modelnet]')
    parser.add_argument('--tta_dataset_path', default='/content/Pointnet_Pointnet2_pytorch/modelnet40_c/modelnet40_c', help='/content/Pointnet_Pointnet2_pytorch/modelnet40_c/modelnet40_c')
    parser.add_argument('--severity', default=5, help='severity for corruption dataset')
    parser.add_argument('--online', default=True, help='online training setting')
    parser.add_argument('--grad_steps', default=1, help='if we train online, we have to set this to one')
    parser.add_argument('--split', type=str, default='test', help='Data split to use: train/test/val')
    parser.add_argument('--debug', action='store_true', help='Use debug mode with a small dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset during loading')
    parser.add_argument('--disable_bn_adaptation', action='store_true', help='Disable batch normalization adaptation')
    parser.add_argument('--stride_step', type=int, default=1, help='Stride step for logging or operations')
    parser.add_argument('--batch_size_tta', type=int, default=1, help='batch size in training')

    return parser.parse_args()

def load_tta_dataset(args):
    # we have 3 choices - every tta_loader returns only point and labels
    root = args.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = '/content/Pointnet_Pointnet2_pytorch/modelnet40_c/modelnet40_c'


        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for ---- is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader








def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    corruptions = [
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
        'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
        'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]

    dataset_name = args.dataset_name
    npoints = args.num_point
    num_class = args.num_category
    level = [5]

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    teacher_model = model.get_model(num_class, normal_channel=args.use_normals)
    student_model = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()

    if not args.use_cpu:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
        criterion = criterion.cuda()

    teacher_model.apply(inplace_relu)
    student_model.apply(inplace_relu)

    # Load weights into both teacher and student models
    model_path = "/content/Pointnet_Pointnet2_pytorch/pretrained_model/modelnet_jt.pth" 
    checkpoint = torch.load(model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.load_state_dict(checkpoint['model_state_dict'])

    # Freezing Teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False

    for param in student_model.parameters():
        param.requires_grad = True

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)

    severity_final_accuracy = {}

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                continue

            tta_loader = load_tta_dataset(args)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []

            for idx, (data, labels) in enumerate(tta_loader):
                student_model.zero_grad()
                student_model.train()

                # Ensurring batch norm layers are disabled for both models
                for m in student_model.modules():
                    if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                        m.eval()  

                teacher_model.eval()  

                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = points.permute(0, 2, 1)  # Now the shape will be [1, 3, 1024]

                        # Forward pass for teacher and student models
                        pred_teacher, _ = teacher_model(points)
                        pseudo_labels = pred_teacher.argmax(dim=1)
                        log_string(f'pseudo labels {pseudo_labels} labels {labels}')

                        print()
                        pred, trans_feat = student_model(points)
                        loss = criterion(pred, pseudo_labels.long(), trans_feat)
                        loss = loss.mean()

                        # Backward pass and optimization step
                        loss.backward()
                        optimizer.step()

                        student_model.zero_grad()
                        optimizer.zero_grad()

                        log_string(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                   f'GradStep - {grad_step} / {args.grad_steps},'
                                   f'Reconstruction Loss {loss.item()}')

                # Inference on the sample
                student_model.eval()
                points = data.cuda()
                points = points.permute(0, 2, 1)
                labels = labels.cuda()

                logits, trans_feat = student_model(points)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)
                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                    log_string(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n')

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)
            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n')
            severity_final_accuracy[args.corruption] = acc

    # Print final accuracy of all corruption kinds
    log_string('------------------------------train finished -----------------------------')
    for _, args.corruption in enumerate(corruptions):
        log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {severity_final_accuracy[args.corruption]} ########\n\n')
        final_mean_accuracy = np.mean(list(severity_final_accuracy.values()))
        log_string(f' mean accuracy {final_mean_accuracy}')



if __name__ == '__main__':
    args = parse_args()
    main(args)
