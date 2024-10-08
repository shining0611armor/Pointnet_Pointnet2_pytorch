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



    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    save_dir = '/content/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_cls_ssg/checkpoints'
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (save_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')



    corruptions = [
    # 'clean',
    'uniform',
    'gaussian',
    'background', 'impulse', 'upsampling',
    'distortion_rbf', 'distortion_rbf_inv', 'density',
    'density_inc', 'shear', 'rotation',
    'cutout',
    'distortion', 'occlusion', 'lidar',
    # 'mixed_corruptions_2_0', 'mixed_corruptions_2_1', 'mixed_corruptions_2_2'
    ]

    dataset_name = args.dataset_name
    npoints = args.num_point
    num_class = args.num_category

    #logger = get_logger(args.log_name)            111111111111111111111111111111
    level = [5]
    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')
                '''
                if corr_id == 0:  # for saving results for easy copying to google sheet
                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
                f_write.write(f'Corruption LEVEL: {args.severity}' + '\n\n')
                '''
            tta_loader = load_tta_dataset(args)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []
            if args.online:
                '''MODEL LOADING'''
                num_class = args.num_category
                model = importlib.import_module(args.model)
                shutil.copy('./models/%s.py' % args.model, str(exp_dir))
                shutil.copy('models/pointnet2_utils.py', str(exp_dir))
                shutil.copy('./train_classification.py', str(exp_dir))

                classifier = model.get_model(num_class, normal_channel=args.use_normals)
                criterion = model.get_loss()
                classifier.apply(inplace_relu)



                args.grad_steps = 1
                model = importlib.import_module(args.model)
                base_model = model.get_model(num_class, normal_channel=args.use_normals)
                
                if not args.use_cpu:
                    base_model = base_model.cuda()
                    criterion = criterion.cuda()

                criterion = model.get_loss()
                base_model.apply(inplace_relu)

                if not args.use_cpu:
                    base_model = base_model.cuda()
                    criterion = criterion.cuda()

                if args.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(
                        base_model.parameters(),
                        lr=args.learning_rate,
                        #betas=(0.9, 0.999),
                        #eps=1e-08,
                        #weight_decay=args.decay_rate
                    )
                else:
                    optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9)

            for idx, (data, labels) in enumerate(tta_loader):
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        #points = misc.fps(points, npoints)
                    else:
                        raise NotImplementedError
                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        #points = [points for _ in range(args.batch_size_tta)]
                        #points = torch.squeeze(torch.vstack(points))
                        '''
                        points = points.data.numpy()
                        points = provider.random_point_dropout(points)
                        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                        points = torch.Tensor(points)
                        points = points.transpose(2, 1)
                        '''
                        if not args.use_cpu:
                            points, target = points.cuda(), labels.cuda()
                        print(points.shape,'1')
                        pred, trans_feat = base_model(points)
                        loss = criterion(pred, labels.long(), trans_feat)
                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        base_model.zero_grad()
                        optimizer.zero_grad()
                    else:
                        continue
                    '''
                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    '''
                    

                    log_string(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                f'GradStep - {grad_step} / {args.grad_steps},'
                                f'Reconstruction Loss {[l for l in loss.val()]}',
                                logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                #points = misc.fps(points, npoints)
                logits, trans_feat = base_model(points)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    log_string(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

                    acc_avg.append(acc.cpu())
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)
            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
 



if __name__ == '__main__':
    args = parse_args()
    main(args)
