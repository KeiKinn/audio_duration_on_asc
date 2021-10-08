import os
import logging
import torch
import time
from utils import create_folder


def create_logging(log_dir, log_name='default'):
    create_folder(log_dir)
    filemode = 'w'

    if log_name == 'default':
        i1 = 0
        while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
            i1 += 1
        log_path = os.path.join(log_dir, '%04d.log' % i1)
    else:
        log_name = log_name + '.log'
        log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def logging_something(args):
    logging.info(args)


def logging_device(device):
    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU.')

    logging.info('GPU number: {}'.format(torch.cuda.device_count()))
    logging.info('-' * 30)


def logging_progress(epoch, epochs, learning_rate):
    logging.info('*' * 10)
    logging.info('Epoch {}/{}, lr:{}'.format(epoch, epochs - 1, learning_rate))
    logging.info('*' * 10)


def logging_process_completed(time_begin):
    time_elapsed = time.time() - time_begin
    logging.info('*' * 10)
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def logging_train_results(phase, accuracy, epoch_loss):
    logging.info('{} Accuracy: {:.4f} loss: {:.4f}'.format(phase, accuracy, epoch_loss))


def logging_augmentation(alpha):
    if alpha > 0.0:
        logging.info("Enable Augmentation: alpha = {}".format(alpha))
    else:
        logging.info("No Augmentation")
    

def logging_deformable(deformable, backbone):
    if deformable:
        logging.info('building DCN v2 on {}'.format(backbone))
    else:
        logging.info('building {}'.format(backbone))