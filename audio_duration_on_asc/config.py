import torch

num_workers = 2
epochs = 50

sample_rate = 44100
sample_len = 10

threshold = 67.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

databases ={

    'dev': "dacase2020_subtask_adth_dev_features.hdf5",
    'train': "dacase2020_subtask_adth_train_features.hdf5",
    'eva': "dacase2020_subtask_adth_eva_features.hdf5",
    'test': "dacase2020_subtask_adth_test_features.hdf5"
}

# baseline
databases_b ={

    'dev': "dacase2020_subtask_baseline_dev_features.hdf5",
    'train': "dacase2020_subtask_baseline_train_features.hdf5",
    'eva': "dacase2020_subtask_baseline_eva_features.hdf5",
    'test': "dacase2020_subtask_baseline_test_features.hdf5"
}

dimension = {
    1.0 : 256,
    4.0 : 128 ,
    5.0 : 128 ,
    5.2 : 128 ,
    5.4 : 128 ,
    5.6 : 128 ,
    5.8 : 128 ,
    6.0 : 128 ,
    6.2 : 128 ,
    6.4 : 128 ,
    6.6 : 128 ,
    6.8 : 256 ,
    7.0 : 256 ,
    7.2 : 256 ,
    7.4 : 256 ,
    7.6 : 256 ,
    7.8 : 256 ,
    8.0 : 256 ,
    8.2 : 256 ,
    8.4 : 256 ,
    8.6 : 256 ,
    8.8 : 256 ,
    9.0 : 256 ,
    9.2 : 256 ,
    9.4 : 256 ,
    9.6 : 256 ,
    9.8 : 256 ,
    10.0 : 384 ,
}