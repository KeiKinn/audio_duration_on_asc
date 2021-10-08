import numpy as np
import sklearn.metrics as metrics
import logging_functions as lf


def read_result_data(data_path):
    output = np.load(data_path)
    return output


def result_report(output, truth, predicts, title='result_report', mode='log'):
    report = metrics.classification_report(output[truth], output[predicts], digits=4)
    if 'log' in mode:
        lf.logging_something('{}:\n {}'.format(title, report))
    else:
        print('{}:\n {}'.format(title, report))


if __name__=="__main__":
    saved_data_dir = '2021-08-24-19-02-26_bs_16_lr_0.001_p_train_deformable'
    data_path = '../../../../nas/student/gPhD_Xin/workspace/dcase2020/saved_data/2021-08-30-11-53-33_BS_16_LR_0.0001_PF_t_BB_exte/saved_data_test.npz'
    output = read_result_data(data_path)
    print(output['truth'])
    result_report(output, 'truth', 'predicts', mode='print')