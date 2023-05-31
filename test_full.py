import os
from datetime import datetime
import torch
import torchvision
import torch.backends.cudnn as cudnn
import pdb
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import cv2
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import scipy.io as sio

########created by you  modelv12_1_test   _scft  _h_all  modelv12_2_nodwt_test    _3layer  modelv12_2_h_all_test  _2_h_all

from modelv12_2_h_all import WavePNet
from data import Datain_testfull as Datain
from mylib import to_var, display_img
from quality_assessment import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_ssim
from args_parser import args_parser

args = args_parser()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# WV4 time_mean:0:00:00.081967

# args.data_path_mat_test = 'data/WV4_453mat/test' #QB_allmat  WV4_453mat WV2_allmat
# # args.bands = 8
# args.model_path='model_fixdwt_cc'

# args.model_path='model_igass'
# args.result_path_full='result_full-30'
def main():
    print(args.model_path)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    use_cuda = torch.cuda.is_available()

    # ----------------------- load data   ---------------------------------------------------------------
    print('load data....')
    '''test '''
    data_test = Datain(args.data_path_mat_test, args.scale_ratio)

    # ----------------------- load model   ---------------------------------------------------------------
    my_model = WavePNet(wavename=args.wavename,
                        bands=args.bands)
    # (self, wavename='haar', bands=4, c1=32, c2=32, c11=16, c21=16) bands
    if use_cuda:
        my_model.cuda()
        print("use cuda")  # qu ma

    # -----------------------Load the trained model parameters -------------------------------------------
    print("Load the trained model parameters ")
    model_path = args.model_path + '/model_epoch1999.pth'
    if os.path.exists(model_path):
        my_model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))

    my_model.eval()
    rmse_list = []
    psnr_list = []
    ergas_list = []
    sam_list = []

    time_list = []

    for i in range(len(data_test)):
        print(
            '_____________________________________________________test_Epoch_{}: ___________________________________'
                .format(i))

        pan_test, ms_test = data_test[i]
        '''add one channel as batch'''
        pan_test = torch.unsqueeze(pan_test, dim=0)
        ms_test = torch.unsqueeze(ms_test, dim=0)

        print('shape of pan ms : ', pan_test.shape, ms_test.shape)

        ms_test_up = F.interpolate(ms_test, scale_factor=args.scale_ratio, mode="bicubic")
        print("change size of ms")
        print(ms_test_up.shape)

        with torch.no_grad():
            # Set mini-batch dataset

            ms = to_var(ms_test_up).detach()
            pan = to_var(pan_test).detach()
            ms_save = to_var(ms_test).detach()

            begin_time = datetime.now()

            out, _, _, _, _, _, _, _, _, _, _, _ = my_model(pan, ms)  #, _, _, _, _ , _, _, _, _


            #out, _, _, _, _, _ = my_model(pan, ms)

            end_time = datetime.now()
            run_time = end_time - begin_time
            print('iter_run_time_{}: '.format(run_time))
            time_list.append(run_time)

            out = out.float().detach().cpu().numpy()

        end_time = datetime.now()
        run_time = end_time - begin_time
        print('iter_run_time_{}: '.format(run_time))

        ###pavia
        pred = np.squeeze(out).transpose(1, 2, 0)

        pan = np.squeeze(pan.detach().cpu().numpy())
        ms = np.squeeze(ms.detach().cpu().numpy()).transpose(1, 2, 0)
        ms_save = np.squeeze(ms_save.detach().cpu().numpy()).transpose(1, 2, 0)

        if not os.path.exists(args.result_path_full):
            os.mkdir(args.result_path_full)
        sio.savemat(args.result_path_full + '/res_%d.mat' % i, {'pred': pred, 'pan': pan, 'ms': ms_save,
                                                                'ms_up': ms})
        print("保存mat完成")

    rmse_mean = np.mean(rmse_list)
    psnr_mean = np.mean(psnr_list)
    ergas_mean = np.mean(ergas_list)
    sam_mean = np.mean(sam_list)

    rmse_var = np.var(rmse_list)
    psnr_var = np.var(psnr_list)
    ergas_var = np.var(ergas_list)
    sam_var = np.var(sam_list)
    print('rmse_mean:{}          rmse_var:{} '.format(rmse_mean, rmse_var))
    print('psnr_mean:{}          psnr_var:{} '.format(psnr_mean, psnr_var))
    print('ergas_mean:{}         ergas_var:{} '.format(ergas_mean, ergas_var))
    print('sam_mean:{}           sam_var:{} '.format(sam_mean, sam_var))

    time_mean = np.mean(time_list)
    #time_var = np.var(time_list)
    print('time_mean:{} '.format(time_mean))

    with open('test.txt', 'a') as f:
        f.write('time:' + str(datetime.now()) + '\n')
        f.write('model_path' + str(model_path) + '\n')
        f.write(
            'rmse_mean:' + str(rmse_mean) + '  ,  ' + 'rmse_var:' + str(rmse_var) + '\n' + 'psnr_mean:' + str(psnr_mean)
            + '  ,  ' + 'psnr_var:' + str(psnr_var) + '\n' + 'ergas_mean:' + str(ergas_mean) + '  ,  ' + 'ergas_var:' +
            str(ergas_var) + '\n' + 'sam_mean:' + str(sam_mean) + '  ,  ' + 'sam_var:' + str(sam_var) + 'time_mean:' +
            str(time_mean) + '\n' + '\n' + '\n' + '\n')
    f.close()


if __name__ == '__main__':
    main()
