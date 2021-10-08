import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
import time
import config as c
import itertools
import plot_functions as pf
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
    backbone = args.backbone
    profile = args.profile
    database = args.dataset
    batch_size = args.batch_size
    train_dataset, eva_dataset, _ = ut.get_dataset(backbone, database, profile)
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=c.num_workers),
        'val': torch.utils.data.DataLoader(eva_dataset, batch_size=batch_size, shuffle=False, num_workers=c.num_workers)
    }
    return dataloaders


def initial_network(args, device, pretrained_dir):
    learning_rate = args.learning_rate
    backbone = args.backbone
    pretrain = args.pretrain
    pretrained_model_name = args.pretrained_model_name
    pretrained_path = pretrained_dir + '/' + pretrained_model_name
    alpha = args.augmentation_alpha
    slices = args.slice
    
    lf.logging_device(device.type)
    lf.logging_augmentation(alpha)

    model = mf.get_train_model(backbone, pretrain, pretrained_path, slices)
    model = mf.set_model(model, device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    scheduler = None

    lf.logging_something("Trainable parameters sum in the model: {}".format(mf.count_pars(model)))

    return model, criterion, optimizer, scheduler

def train(args, paths, dataloaders, model, criterion, optimizer, scheduler):
    backbone = args.backbone
    epochs = args.epochs
    alpha = args.augmentation_alpha
    slices = args.slice
    
    img_dir_path = paths['img_dir_path']
    storage_dir_path = paths['storage_dir_path']
    models_dir_path = paths['models_dir_path']

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    time_begin = time.time()
    for epoch in range(epochs):
        lf.logging_progress(epoch, epochs, optimizer.param_groups[0]['lr'])

        y_scores_npy = []
        predicts_npy = []
        truth_npy = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            predicts = []
            truth = []
            y_scores = []
            total = 0
            correct = 0
            
            for idx, batch_data in enumerate(dataloaders[phase]):

                inputs, labels, labels_b, lam = ut.get_train_data(backbone, batch_data, device, alpha, phase, slices=slices)

                with torch.set_grad_enabled('train' in phase):

                    outputs = model(inputs)

                    loss_func = ut.mixup_criterion(labels, labels_b, lam)
                    loss = loss_func(criterion, outputs)
                    _, preds = torch.max(outputs, dim=1)

                    if 'train' in phase:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    y_scores.append(outputs.tolist())
                    predicts.append(preds.tolist())
                    truth.append(labels.tolist())

                    total += labels.size(0)
                    correct += lam * (preds == labels).sum().item() + (1 - lam) * (preds == labels_b).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            accuracy = 100 * correct / total
            lf.logging_train_results(phase, accuracy, epoch_loss)

            if 'train' in phase:
                train_loss.append(epoch_loss)
                train_accuracy.append(accuracy)
            else:
                val_loss.append(epoch_loss)
                val_accuracy.append(accuracy)

            y_scores = list(itertools.chain(*y_scores))
            predicts = list(itertools.chain(*predicts))
            truth = list(itertools.chain(*truth))

            y_scores_npy.append(y_scores)
            predicts_npy.append(predicts)
            truth_npy.append(truth)

            # Plot
            if phase == 'val':
                if epoch % 20 == 19 or epoch == (epochs-1) or accuracy > c.threshold:
                    pf.loss_plot(train_loss, val_loss, img_dir_path, epoch)
                    pf.accuracy_plot(train_accuracy, val_accuracy, img_dir_path, epoch)

                if epoch % 20 == 19 or epoch == (epochs-1) or accuracy > c.threshold:
                    storage_path = ut.generate_saved_data_file_info(storage_dir_path, epoch)
                    ut.save_result(storage_path, y_scores_npy, predicts_npy, truth_npy)
                    mf.save_model(model, optimizer, models_dir_path, epoch)

                    output = oa.read_result_data(storage_path)
                    oa.result_report(output, 'truth_val', 'predicts_val', 'val report')
                    oa.result_report(output, 'truth_train', 'predicts_train', 'train report')

    lf.logging_process_completed(time_begin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    ## settings
    parser_train.add_argument('--dataset', type=str, default='../../dataset')
    parser_train.add_argument('--workspace', type=str, default='../workspace')
    parser_train.add_argument('--profile', type=str, choices=['train', 'dev'])
    ## model settings
    parser_train.add_argument('--backbone', type=str, default='baseline')
    parser_train.add_argument('--pretrain', action='store_true')
    parser_train.add_argument('--pretrain_path', type=str, default=None)
    parser_train.add_argument('--pretrained_model_name', type=str, default=None)
    ## training
    parser_train.add_argument('--learning_rate', type=float, default=0.001)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=30)
    parser_train.add_argument('--augmentation_alpha', type=float, default=1.0)
    parser_train.add_argument('--slice', type=float, default=0.0)

    # Parse arguments
    args = parser.parse_args()
    if args.mode:
        paths, device = initialization(args)
        model, criterion, optimizer, scheduler = initial_network(args, device, paths['models_dir_path'])
        dataloaders = running_profile(args)

        train(args, paths, dataloaders, model, criterion, optimizer, scheduler)
    else:
        raise Exception('Not set mode')
