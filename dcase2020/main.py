import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
import time
import constants as c
import itertools
import plot_functions as pf
import utils as ut
import model_functions as mf
import logging_functions as lf
import output_analysis as oa
import Controller as C
import BasicConfigController as BC
import BasicConfigData as BCD
import BoardController as BoC


def initialization():
    generals_str = ctrl.generals_str

    logs_dir = os.path.join(ctrl.workspace, 'logs')
    lf.create_logging(logs_dir, generals_str)
    lf.logging_something("tag: {}".format(generals_str))
    lf.logging_something(args)
    lf.logging_device(ctrl.device.type)
    lf.logging_augmentation(ctrl.alpha)


def running_profile():
    train_dataset, eva_dataset, _ = ut.get_dataset(ctrl.backbone, ctrl.database, ctrl.profile)
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=ctrl.batch_size, shuffle=True,
                                             num_workers=c.num_workers),
        'val': torch.utils.data.DataLoader(eva_dataset, batch_size=ctrl.batch_size, shuffle=False,
                                           num_workers=c.num_workers)
    }
    return dataloaders


def initial_network():
    pretrained_path = ctrl.pretrained_model_path()

    model = mf.get_train_model(ctrl.backbone, ctrl.pretrain, pretrained_path, ctrl.slices)
    model = mf.set_model(model, ctrl.device)

    criterion = nn.CrossEntropyLoss().to(ctrl.device)

    optimizer = optim.Adam(model.parameters(), lr=ctrl.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.,
                           amsgrad=True)

    scheduler = None

    return model, criterion, optimizer, scheduler


def train(dataloaders, model, criterion, optimizer, scheduler):

    epochs = ctrl.epochs
    img_dir_path = ctrl.folder_in_workspace('img')
    storage_dir_path = ctrl.folder_in_workspace('storage')
    models_dir_path = ctrl.folder_in_workspace('models')

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    time_begin = time.time()
    for epoch in range(epochs):
        lf.logging_progress(epoch, epochs, optimizer.param_groups[0]['lr'])
        if basic_cfg.is_updated():
            basic_cfg_data.update_values(basic_cfg.values)
            lf.logging_something("New Basic Config is loaded...")

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

                inputs, labels, labels_b, lam = ut.get_train_data(ctrl.backbone, batch_data, ctrl.device, ctrl.alpha, phase,
                                                                  slices=ctrl.slices)

                with torch.set_grad_enabled('train' == phase):

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

                BoC.writer.add_scalars("loss", {"train_loss": train_loss[-1],
                                               "val_loss": val_loss[-1]}, epoch)
                BoC.writer.add_scalars("accuracy", {"train_accuracy": train_accuracy[-1],
                                               "val_accuracy": val_accuracy[-1]}, epoch)

            y_scores = list(itertools.chain(*y_scores))
            predicts = list(itertools.chain(*predicts))
            truth = list(itertools.chain(*truth))

            y_scores_npy.append(y_scores)
            predicts_npy.append(predicts)
            truth_npy.append(truth)

            # Plot
            if phase == 'val':
                if epoch % basic_cfg_data.period == (basic_cfg_data.period - 1) or epoch == (epochs - 1) or accuracy > basic_cfg_data.threshold:
                    pf.loss_plot(train_loss, val_loss, img_dir_path, epoch)
                    pf.accuracy_plot(train_accuracy, val_accuracy, img_dir_path, epoch)

                    mf.save_model(model, optimizer, models_dir_path, epoch)

                    storage_path = ut.generate_saved_data_file_info(storage_dir_path, epoch)
                    ut.save_result(storage_path, y_scores_npy, predicts_npy, truth_npy, basic_cfg_data.is_data_save)

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
    parser_train.add_argument('--profile', type=str, choices=['train', 'dev'], default='train')
    ## model settings
    parser_train.add_argument('--backbone', type=str, default='baseline')
    parser_train.add_argument('--pretrain', action='store_true')
    parser_train.add_argument('--pretrain_path', type=str, default=None)
    parser_train.add_argument('--pretrained_model_name', type=str, default=None)
    ## training
    parser_train.add_argument('--learning_rate', type=float, default=0.001)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--epochs', type=int, default=30)
    parser_train.add_argument('--augmentation_alpha', type=float, default=-1.0)
    parser_train.add_argument('--slice', type=float, default=0.0)

    # Parse arguments
    args = parser.parse_args()
    if args.mode:
        ctrl = C.Controller(args)
        basic_cfg = BC.BasicConfigController("../workspace/basic_cfg.json")
        basic_cfg_data = BCD.BasicConfigData(**basic_cfg.values)
        initialization()
        model, criterion, optimizer, scheduler = initial_network()
        dataloaders = running_profile()

        train(dataloaders, model, criterion, optimizer, scheduler)
    else:
        raise Exception('Not set mode')
