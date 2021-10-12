import os
import torch
import time
import utils as ut


class Controller(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.workspace = args.workspace
        self.pretrain = args.pretrain
        if self.pretrain:
            self.pretrained_path = args.pretrain_path

        self.backbone = args.backbone
        self.profile = args.profile
        self.database = args.dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.learning_rate = args.learning_rate
        self.pretrained_model_name = args.pretrained_model_name
        self.alpha = args.augmentation_alpha
        self.slices = args.slice

        self.generals_str = self.__generals_str()
        self.paths = self.__create_workspace()

    def __generals_str(self):
        if self.pretrain:
            generals_str = self.pretrained_path
        else:
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            train_info = '_BS_' + str(self.batch_size) + '_LR_' + str(self.learning_rate) \
                         + '_PF_' + self.profile[0]
            model_info = '_BB_' + self.backbone[0:4]
            if self.alpha > 0.0:
                augmentation = ('_AA_' + str(self.alpha))
            else:
                augmentation = ''

            slices_info = ''
            if self.slices > 0.0:
                slices_info = '_SL_' + str(self.slices)

            tag = train_info + model_info + augmentation + slices_info
            generals_str = time_stamp + tag

        return generals_str

    def __create_workspace(self):
        paths = dict()
        folder_name = self.generals_str
        paths['img'] = os.path.join(self.workspace, 'img', '{}'.format(folder_name))
        paths['storage'] = os.path.join(self.workspace, 'saved_data', '{}'.format(folder_name))
        paths['models'] = os.path.join(self.workspace, 'models', '{}'.format(folder_name))

        for key in paths:
            ut.create_folder(paths[key])
        return paths

    def workspace(self):
        return self.paths

    def folder_in_workspace(self, folder_name):
        return self.paths[folder_name]
