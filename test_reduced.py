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
########created by you
from modelv12_2_h_all import WavePNet
from data import Datain
from mylib import to_var, display_img
#from quality_assessment import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_ssim
from args_parser import args_parser

args = args_parser()
# wv4
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#  WV4_453mat WV2_allmat  QB_allmat
# args.data_path_mat_test = 'data/WV2_allmat/test'
# args.bands = 8
# args.model_path='model_igass'
# args.model_path='model-50'
# args.result_path='result-50'
# args.data_path_mat_test = 'data/WV4_453mat/test'
# args.bands = 4
# args.model_path='model_dwt_cc'
def main():
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    use_cuda = torch.cuda.is_available()

    # ----------------------- load data   ---------------------------------------------------------------
    print('load data....')
    '''test '''
    data_test = Datain(args.data_path_mat_test, args.scale_ratio)
    #data_test = Datain('data/WV2_453mat/test', args.scale_ratio)#args.data_path_mat_test _testfull

    # test_loader = torch.utils.data.DataLoader(data_test,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=True,
    #                                            num_workers=1)

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


        ref_test, pan_test, ms_test = data_test[i]
        '''add one channel as batch'''
        pan_test = torch.unsqueeze(pan_test, dim=0)
        ms_test = torch.unsqueeze(ms_test, dim=0)
        ref_test = torch.unsqueeze(ref_test, dim=0)
        print('shape of pan ms ref : ', pan_test.shape, ms_test.shape, ref_test.shape)

        ms_test_up = F.interpolate(ms_test, scale_factor=args.scale_ratio, mode="bicubic")
        print("change size of ms")
        print(ms_test_up.shape)

        with torch.no_grad():
            # Set mini-batch dataset
            ref = to_var(ref_test).detach()
            ms = to_var(ms_test_up).detach()
            pan = to_var(pan_test).detach()
            ms_save = to_var(ms_test).detach()

            begin_time = datetime.now()

            out, _, _, _, _, _, _, _, _, _, _, _ = my_model(pan, ms) #, _, _, _, _, _, _, _, _

            #out, _, _, _, _, _= my_model(pan, ms)

            end_time = datetime.now()
            run_time = end_time - begin_time
            print('iter_run_time_{}: '.format(run_time))
            time_list.append(run_time)

            ref = ref.float().detach().cpu().numpy()
            out = out.float().detach().cpu().numpy()

            rmse = calc_rmse(ref, out)
            psnr = calc_psnr(ref, out)
            ergas = calc_ergas(ref, out)
            sam = calc_sam(ref, out)
            # ssim = calc_ssim(ref, out)

            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ergas_list.append(ergas)
            sam_list.append(sam)
            print('RMSE:   {:.4f};'.format(rmse))
            print('PSNR:   {:.4f};'.format(psnr))
            print('ERGAS:   {:.4f};'.format(ergas))
            print('SAM:   {:.4f}.'.format(sam))
            with open('test.txt', 'a') as f:
                f.write('i:' + str(i) + '  ,  ' + 'rmse:' + str(rmse) + '  ,  ' + 'psnr:' + str(psnr) + '  ,  '
                        + 'ergas:' + str(ergas) + '  ,  ' + 'sam:' + str(sam) + '\n')  # + str(ssim) + ','
        f.close()



        pred = np.squeeze(out).transpose(1, 2, 0)
        ref = np.squeeze(ref).transpose(1, 2, 0)

        # print("_______________________________________________pan shape")
        # print(pan.shape)
        # print(np.squeeze(pan).shape)
        pan = np.squeeze(pan.detach().cpu().numpy())
        ms = np.squeeze(ms.detach().cpu().numpy()).transpose(1, 2, 0)
        ms_save = np.squeeze(ms_save.detach().cpu().numpy()).transpose(1, 2, 0)

        # '''cause 1023.5'''
        # pred = pred * 1023.5 + 1023.5
        # ref = ref * 1023.5 + 1023.5
        # pan = pan * 1023.5 + 1023.5
        # ms = ms * 1023.5 + 1023.5
        # ms_save = ms_save * 1023.5 + 1023.5

        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
        sio.savemat(args.result_path + '/res_%d.mat' % i, {'pred': pred, 'ref': ref, 'pan': pan, 'ms': ms_save,
                                                           'ms_up': ms})
        print(" Save mat success!")


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
    print('time_mean:{}'.format(time_mean))
    #time_var = np.var(time_list)
    #print('time_mean:{}          time_var:{} '.format(time_mean, time_var))


    with open('test.txt', 'a') as f:
        f.write('time:' + str(datetime.now()) + '\n')
        f.write('model_path' + str(model_path) + '\n')
        # f.write(
        #     'rmse_mean:' + str(rmse_mean) + '  ,  ' + 'rmse_var:' + str(rmse_var) + '\n' + 'psnr_mean:' + str(psnr_mean)
        #     + '  ,  ' + 'psnr_var:' + str(psnr_var) + '\n' + 'ergas_mean:' + str(ergas_mean) + '  ,  ' + 'ergas_var:' +
        #     str(ergas_var) + '\n' + 'sam_mean:' + str(sam_mean) + '  ,  ' + 'sam_var:' + str(sam_var) + 'time_mean:' +
        #     str(time_mean) + '  ,  ' + 'time_var:' + str(time_var) + '\n' + '\n' + '\n' + '\n')
        f.write(
            'rmse_mean:' + str(rmse_mean) + '  ,  ' + 'rmse_var:' + str(rmse_var) + '\n' + 'psnr_mean:' + str(psnr_mean)
            + '  ,  ' + 'psnr_var:' + str(psnr_var) + '\n' + 'ergas_mean:' + str(ergas_mean) + '  ,  ' + 'ergas_var:' +
            str(ergas_var) + '\n' + 'sam_mean:' + str(sam_mean) + '  ,  ' + 'sam_var:' + str(sam_var) + 'time_mean:' +
            str(time_mean) + '\n' + '\n' + '\n' + '\n')
    f.close()


if __name__ == '__main__':
    main()
#loss_ref:0.23520179092884064 __loss_h:10.114566802978516 __loss_lh:5.649776935577393 __loss_total:315.5220642089844