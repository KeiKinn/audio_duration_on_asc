import os
import torch
import numpy as np
import DatasetGenerator as dg
import logging
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score
import config as c
import random


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_dataset(backbone, database, profile='train'):

    if "baseline" == backbone:
        database_name = c.databases_b
    elif "dresnet" == backbone:
        database_name = c.databases


    train_hdf5_path = os.path.join(database, 'DCASE2020', database_name[profile])
    val_hdf5_path = os.path.join(database, 'DCASE2020', database_name['eva'])
    test_hdf5_path = os.path.join(database, 'DCASE2020', database_name['test'])
    train_dataset = dg.DCASE2020(hdf5_path=train_hdf5_path, backbone=backbone)
    val_dataset = dg.DCASE2020(hdf5_path=val_hdf5_path, backbone=backbone)
    test_dataset = dg.DCASE2020(hdf5_path=test_hdf5_path, backbone=backbone)
    return train_dataset, val_dataset, test_dataset


def get_train_data(backbone, batch_data, device, alpha, phase, slices=0.0): 
    alpha_temp = -1
    if 'train' in phase:
        alpha_temp = alpha
    logmel = adjust_data(batch_data, device, 'logmel', slices, phase)
    if "baseline" == backbone:
        inputs = logmel
    elif "dresnet" == backbone:
        delta = adjust_data(batch_data, device, 'delta', slices, phase)
        delta_delta = adjust_data(batch_data, device, 'delta-delta', slices, phase)
        inputs = torch.cat([logmel, delta, delta_delta], dim=1)
    labels = batch_data['label'].type(torch.LongTensor).to(device)
    inputs, labels, labels_b, lam = mixup_data(inputs, labels, alpha=alpha_temp, device=device)
    return inputs, labels, labels_b, lam

def adjust_data(batch_data, device, key, slices=0.0, phase='train'):
    result = batch_data[key].to(device)
    result = torch.unsqueeze(result, dim=1)
    result = torch.swapaxes(result, 2, 3)

    if slices > 0.0:
        random_upper_edge = int((1 - slices/c.sample_len) * result.shape[3])
        start_slice = random.randint(0, random_upper_edge) if 'train' == phase else 0
        end_slice = start_slice + int(slices/c.sample_len*result.shape[3])
        result = result[:, :, :, start_slice:end_slice]
    return result


def scoring(truth, pred, y_scores):
    y_scores = np.asarray(y_scores)
    uar = recall_score(truth, pred, average='macro')
    auc = roc_auc_score(truth, y_scores[:, 1], average='macro')
    confusion_mat = confusion_matrix(truth, pred, labels=list(range(2)))
    return confusion_mat, uar, auc


def generate_saved_data_file_info(storage_dir_path, tag):

    while os.path.isfile(storage_dir_path+'/saved_data_{}.npz'.format(tag)):
        tag += 10000
    saved_data_name = 'saved_data_' + str(tag) + '.npz'

    storage_path = os.path.join(storage_dir_path, '{}'.format(saved_data_name))
    return storage_path


def save_result(saved_data_path, y_scores_npy, predicts_npy, truth_npy, mode='train'):
    if 'train' in mode:
        y_scores_train = np.hstack(y_scores_npy[::2])
        y_scores_val = np.hstack(y_scores_npy[1::2])
        predicts_train = np.hstack(predicts_npy[::2])
        predicts_val = np.hstack(predicts_npy[1::2])
        truth_train = np.hstack(truth_npy[::2])
        truth_val = np.hstack(truth_npy[1::2])

        np.savez(saved_data_path,
                y_scores_train=y_scores_train,
                predicts_train=predicts_train,
                truth_train=truth_train,
                y_scores_val=y_scores_val,
                predicts_val=predicts_val,
                truth_val=truth_val)
    elif 'test' in mode:
        np.savez(saved_data_path, 
                y_scores=y_scores_npy,
                predicts=predicts_npy,
                truth=truth_npy)

    logging.info('*' * 10)
    logging.info('Output data with have been saved at {}'.format(saved_data_path))


def mixup_data(x, y, alpha=-1.0, device='cpu'):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)