import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import argparse
import time
import constants as c
import itertools
import utils as ut
import model_functions as mf
import logging_functions as lf
import output_analysis as oa


def initialization(args):
    workspace = args.workspace
    pretrain = args.pretrain
    device = c.device
    if pretrain:
        pretrained_path = args.pretrain_path

    paths = dict()
    logs_dir = os.path.join(workspace, 'logs')
    generals_str = pretrained_path if pretrain else ut.generate_string(args)

    paths['img_dir_path'] = os.path.join(workspace, 'img', '{}'.format(generals_str))
    paths['storage_dir_path'] = os.path.join(workspace, 'saved_data', '{}'.format(generals_str))
    paths['models_dir_path'] = os.path.join(workspace, 'models', '{}'.format(generals_str))

    ut.create_folder(paths['img_dir_path'])
    ut.create_folder(paths['storage_dir_path'])
    ut.create_folder(paths['models_dir_path'])

    lf.create_logging(logs_dir, generals_str)
    lf.logging_something("tag: {}".format(generals_str))
    lf.logging_something(args)
    return paths, device


def running_profile(args):
    database = args.dataset
    batch_size = args.batch_size
    _, _, test_dataset = ut.get_dataset(database, 'test')
    dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=c.num_workers)
    return dataloaders


def initial_network(args, device, pretrained_dir):
    backbone = args.backbone
    deformable = args.deformable
    layers = args.layers
    pretrain = args.pretrain
    pretrained_model_name = args.pretrained_model_name
    pretrained_path = pretrained_dir + '/' + pretrained_model_name
    
    lf.logging_device(device.type)

    model = mf.re_get_train_model(backbone, deformable, layers, pretrain, pretrained_path)
    model = mf.set_model(model, device)

    return model


def test(paths, dataloaders, model):
    
    storage_dir_path = paths['storage_dir_path']


    y_scores_npy = []
    predicts_npy = []
    truth_npy = []
    time_begin = time.time()

    model.eval()

    predicts = []
    truth = []
    y_scores = []
    total = 0
    correct = 0
    for idx, batch_data in enumerate(dataloaders):

        inputs, labels = ut.get_train_data(batch_data, device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)

            y_scores.append(outputs.tolist())
            predicts.append(preds.tolist())
            truth.append(labels.tolist())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    lf.logging_train_results('test', accuracy, 0.0)

    y_scores = list(itertools.chain(*y_scores))
    predicts = list(itertools.chain(*predicts))
    truth = list(itertools.chain(*truth))

        # confusion_mat, uar, auc = ut.scoring(truth, predicts, y_scores)
        # logging.info(
        #     '{}: auc: {:.4f}, uar: {:.4f}, confusion_mat:{}'.format(phase, auc, uar, confusion_mat))
    # scheduler.step()

    # # Plot
    # pf.loss_plot(train_loss, val_loss, img_dir_path, epoch)

    storage_path = ut.generate_saved_data_file_info(storage_dir_path, 'test')
    ut.save_result(storage_path, y_scores_npy, predicts, truth, 'test')

    output = oa.read_result_data(storage_path)
    oa.result_report(output, 'truth', 'predicts', 'test_report')

    lf.logging_process_completed(time_begin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('test')
    ## settings
    parser_train.add_argument('--dataset', type=str, default='../../dataset')
    parser_train.add_argument('--workspace', type=str, default='../workspace')
    ## model settings
    parser_train.add_argument('--backbone', type=str, default='baseline')
    parser_train.add_argument('--deformable', action='store_true')
    parser_train.add_argument('--layers', nargs='*', default=None)
    parser_train.add_argument('--pretrain', action='store_true')
    parser_train.add_argument('--pretrain_path', type=str, default=None)
    parser_train.add_argument('--pretrained_model_name', type=str, default=None)

    parser_train.add_argument('--batch_size', type=int, default=16)

    # Parse arguments
    args = parser.parse_args()
    if args.mode:
        paths, device = initialization(args)
        model = initial_network(args, device, paths['models_dir_path'])
        dataloaders = running_profile(args)

        test(paths, dataloaders, model)

    else:
        raise Exception('Not set mode')
