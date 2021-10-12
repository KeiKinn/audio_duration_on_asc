import os

import torch

import logging_functions as lf
import model as md


# Note: this is easy to transplant
def generate_models_file_info(models_dir_path, tag):
    while os.path.isfile(models_dir_path + '/saved_model_{}.pth'.format(tag)):
        tag += 10000

    models_name = 'saved_model_' + str(tag) + '.pth'
    models_path = os.path.join(models_dir_path, '{}'.format(models_name))
    return models_path


def set_model(model, device):
    model = torch.nn.DataParallel(model)
    model.to(device)
    return model


def save_model(model, optimizer, models_dir_path, tag):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    models_name = generate_models_file_info(models_dir_path, tag)
    torch.save(checkpoint, models_name)
    lf.logging_something('Model with tag {} is saved to {}'.format(tag, models_name))


def get_train_model(backbone=None, pretrain=False, path=None, slices=0.0):
    if backbone == 'baseline':
        model = md.BaselineCNN(1, 10, slices=slices)
    elif backbone == 'dresnet':
        model = md.DualResnet(3, 10, slices=slices)

    if pretrain == True:
        lf.logging_something('loading Pretrained model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = torch.device(device)
        pretrained_path = path
        model.load_state_dict(torch.load(pretrained_path, map_location)['model'])

    lf.logging_something('model\n {}'.format(model))
    lf.logging_something("Trainable parameters sum in the model: {}".format(model.count_pars()))
    return model
