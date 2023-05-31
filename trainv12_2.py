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

from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torchvision.utils import make_grid
import string

########created by you   _fixed    modelv12_2_h_all_test

from modelv12_2_h_all import WavePNet
from data import Datain
from mylib import to_var, loss_func29_h #,  loss_func32 # loss_func29_h
from quality_assessment import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_ssim
from args_parser import args_parser

'''
this trainv2.py  includes val part
datalload part is different from train.py
'''

# SSRNET

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# __________________ set params   __________________________________
args = args_parser()
print(args)
with open('train.txt', 'a') as f:
    f.write('\n' + '\n' + '\n' + 'time:' + str(datetime.now()) + '\n')
    f.write('args:' + str(args) + '\n' + '\n')
f.close()

# args.model_path='model_fixdwt_cc'
# args.data_path_mat_train = 'data/WV4_453mat/train'
# args.data_path_mat_val = 'data/WV4_453mat/val'
# args.bands = 4
def main():
    #  ------------------ make dirs if they are not exit -----------------------------------------------
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # -----------------------             ---------------------------------------------------------------
    writer = SummaryWriter()
    # best_test_loss = np.inf  # infinity
    use_cuda = torch.cuda.is_available()

    # ----------------------- load data   ---------------------------------------------------------------
    print('load data....')
    ''' train part'''
    data_train = Datain(args.data_path_mat_train, args.scale_ratio)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=1)
    '''val part'''
    data_val = Datain(args.data_path_mat_val, args.scale_ratio)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=1)

    # -----------------------  load model   ---------------------------------------------------------------
    my_model = WavePNet(wavename=args.wavename,
                              bands=args.bands)  # (self, wavename='haar', bands=4, c1=32, c2=32, c11=16, c21=16) bands
    if use_cuda:
        my_model.cuda()
        print("use cuda")

    # -----------------------  Loss and Optimizer  ------------------------------------------------------------

    criterion = loss_func29_h().cuda()#loss_func32().cuda()#nn.L1Loss().cuda()##nn.L1Loss().cuda()#loss_func14().cuda()#nn.L1Loss().cuda()# #nn.MSELoss().cuda()#

    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)  # gai
    # optimizer = torch.optim.SGD(my_model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    '''some parameters needed'''
    best_psnr = 0
    best_rmse = np.inf
    print('psnr: ', best_psnr)
    total_loss = 0
    # Epochs
    print('Start Training: ')
    my_model.train()
    loss = np.inf

    '''train start'''
    for epoch in range(args.epochs):
        print(
            '_________________________________________________________________________________________________________'
            '_______________________________________________________________Train_Epoch_{}:__'.format(epoch))
        count = 0
        t1 = datetime.now()

        for batch_id, (ref, pan, ms) in enumerate(train_loader):
            count += 1
            print('Train_Epoch_{}:  ______________Train_iter_{}: '.format(epoch, batch_id))
            '''ms upsample'''
            ms = F.interpolate(ms, scale_factor=args.scale_ratio, mode="bicubic")
            print("change size of ms")
            print(ms.shape)

            if use_cuda:
                pan = Variable(pan).cuda()
                ms = Variable(ms).cuda()
                ref = Variable(ref).cuda()
            # print("in data type")
            # print(type(pan_batch))
            # , quary_h1, key_h1, value_h1
            #out, low, high3, high2, high1, panl_latent, msl_latent, panlh_latent, mslh_latent, panllh_latent, msllh_latent = my_model(pan, ms)
            #out, panl_latent, msl_latent, panl2_latent, msl2_latent = my_model(pan, ms)
            out, pc_1, pgc_1, pc_2, pgc_2, mc_1, mgc_1, mc_2, mgc_2, low, high2, high1, panh_latent, msh_latent, panlh_latent, mslh_latent = my_model(pan, ms)  #
            #out, pc_1, pgc_1, pc_2, pgc_2, mc_1, mgc_1, mc_2, mgc_2, low, high2, high1, panl_latent1, msl_latent1, panl_latent, msl_latent= my_model(pan, ms)
            # pc2 = pc_2.detach().cpu().numpy()
            # mc2 = mc_2.detach().cpu().numpy()
            # print('pc2.shape',pc2.shape,'mc2.shape',mc2.shape)
            # , pc2, mc2
            # , quary_h1, key_h1, value_h1
            loss = criterion(ref, out, panh_latent, msh_latent, panlh_latent, mslh_latent)#, panh_latent, msh_latent, panlh_latent, mslh_latent
            #loss = criterion(ref, out, panl_latent, msl_latent, panl_latent1, msl_latent1) panl_latent1, msl_latent1, panl_latent, msl_latent, panh_latent, msh_latent, panlh_latent, mslh_latent

            print("__________________________________loss________________________________________________")
            print(loss)
            print("__________________________________loss________________________________________________")
            total_loss += loss.item()

            optimizer.zero_grad()  # set gradient=0，以免影响其他batch
            loss.backward()  # 后向传播，计算梯度
            print('后向传播，计算梯度')
            optimizer.step()  # 利用梯度更新w b 参数

        t2 = datetime.now()
        run_time = t2 - t1
        print('____________________________one epoch run_time_{}: '.format(run_time))

        # One epoch's validation
        print('Val_Epoch_{}: '.format(epoch))
        recent_ergas, rmse, psnr_mean, sam_mean = val(my_model, epoch, val_loader, args)  # , ssim
        print('recent_psnr: {}'.format(psnr_mean))

        # # save model

        # is_best = psnr_mean > best_psnr
        # best_psnr = max(psnr_mean, best_psnr)
        # if is_best and best_psnr > 0:
        #     torch.save(my_model.state_dict(), args.model_path + '/model_epoch%d.pth' % epoch)
        #     print('Saved!')
        #     print('')
        #
        # print('best_psnr: ', best_psnr)

        is_best = rmse < best_rmse
        best_rmse = min(rmse, best_rmse)
        if is_best and best_rmse > 0:
            torch.save(my_model.state_dict(), args.model_path + '/model_epoch%d.pth' % epoch)
            print('Saved!')
            print('')

        print('best_rmse: ', best_rmse)

        # print("__________________________________loss.item()________________________________________________")
        # print(loss.item())
        # print(type(loss.item()))
        # print(total_loss)
        total_loss /= count
        print('total iters: ', count)
        print('train epoch [%d/%d] average_loss %.5f' % (epoch, count, total_loss))

        ''' visiualize scalar '''

        # writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=epoch)
        
        writer.add_scalar(tag="total_loss", scalar_value=total_loss, global_step=epoch)
        writer.add_scalar(tag="recent_ergas_mean", scalar_value=recent_ergas, global_step=epoch)
        writer.add_scalar(tag="recent_psnr_mean", scalar_value=psnr_mean, global_step=epoch)
        writer.add_scalar(tag="recent_rmse_mean", scalar_value=rmse, global_step=epoch)
        writer.add_scalar(tag="recent_sam_mean", scalar_value=sam_mean, global_step=epoch)



        writer.add_image('MS', make_grid(to_var(ms[2]).detach().cpu().unsqueeze(dim=1), nrow=4, padding=5), global_step=epoch)
        writer.add_image('PAN', make_grid(to_var(pan[2]).detach().cpu().unsqueeze(dim=1), nrow=1, padding=5), global_step=epoch)
        # writer.add_image('feature_map_pc_1', make_grid(to_var(pc_1[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_pgc_1', make_grid(to_var(pgc_1[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_pc_2', make_grid(to_var(pc_2[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_pgc_2', make_grid(to_var(pgc_2[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_mc_1', make_grid(to_var(mc_1[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_mgc_1', make_grid(to_var(mgc_1[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_mc_2', make_grid(to_var(mc_2[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('feature_map_mgc_2', make_grid(to_var(mgc_2[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('low', make_grid(to_var(low[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('high2', make_grid(to_var(high2[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        # writer.add_image('high1', make_grid(to_var(high1[2]).detach().cpu().unsqueeze(dim=1), nrow=8, padding=5), global_step=epoch)
        writer.add_image('out', make_grid(to_var(out[2]).detach().cpu().unsqueeze(dim=1), nrow=4, padding=5), global_step=epoch)
        
        # writer.add_image('feature_map_pc_1', make_grid([pc_1[2].detach().cpu().unsqueeze(dim=1), pc_1[17].detach().cpu().unsqueeze(dim=1), pc_1[25].detach().cpu().unsqueeze(dim=1)], padding=5, normalize=True, scale_each=True, pad_value=1), global_step=epoch)
        # writer.add_image('feature_map_pc_1', make_grid([pc_1[2].detach().cpu().unsqueeze(dim=1), pc_1[17].detach().cpu().unsqueeze(dim=1), pc_1[25].detach().cpu().unsqueeze(dim=1)], padding=5, normalize=True, scale_each=True, pad_value=1), global_step=epoch)
        # writer.add_image('feature_map_pc_1', make_grid([pc_1[2].detach().cpu().unsqueeze(dim=1), pc_1[17].detach().cpu().unsqueeze(dim=1), pc_1[25].detach().cpu().unsqueeze(dim=1)], padding=5, normalize=True, scale_each=True, pad_value=1), global_step=epoch)
        # writer.add_scalar(tag="recent_ssim", scalar_value=ssim, global_step=epoch)

        # writer.add_scalars('loss/scalar_group', {"loss": epoch * loss,  "total_loss": epoch * total_loss})
        # writer.add_graph(fcn_model, input_to_model)#和上面input_to_model=torch.randn((1, 3, 320, 320)).cuda()一起用于打印网络结构

        # if epoch == 400:
        #     args.learning_rate *= 0.1
        #     optimizer.param_groups[0]['lr'] = args.learning_rate
        #     with open('train.txt', 'a') as f:
        #         f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
        #             args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
        #     f.close()
        # if epoch == 200:
        #     args.learning_rate *= 0.1
        #     optimizer.param_groups[0]['lr'] = args.learning_rate
        #     with open('train.txt', 'a') as f:
        #         f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
        #             args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
        #     f.close()
        # if epoch == 1200:
        #     args.learning_rate *= 0.1
        #     optimizer.param_groups[0]['lr'] = args.learning_rate
        #     with open('train.txt', 'a') as f:
        #         f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
        #             args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
        #     f.close()
        # if epoch == 1500:
        #     args.learning_rate *= 0.5
        #     optimizer.param_groups[0]['lr'] = args.learning_rate
        #     with open('train.txt', 'a') as f:
        #         f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
        #             args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
        #     f.close()
        #
        # if epoch == 200:
            # args.learning_rate *= 0.1
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 400:
            # args.learning_rate *= 0.1
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
            
        # if epoch == 600:
            # args.learning_rate *= 0.1
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        
        # if epoch == 100:
            # args.learning_rate = 0.0002
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 200:
            # args.learning_rate = 0.0001
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 300:
            # args.learning_rate *= 0.5
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 400:
            # args.learning_rate *= 0.2
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 100:
            # args.learning_rate *= 0.1
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()
        # if epoch == 200:
            # args.learning_rate *= 0.1
            # optimizer.param_groups[0]['lr'] = args.learning_rate
            # with open('train.txt', 'a') as f:
                # f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
                    # args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
            # f.close()

        # while not (epoch % 100):
        #     args.learning_rate *= 0.1
        #     optimizer.param_groups[0]['lr'] = args.learning_rate
        #     with open('train.txt', 'a') as f:
        #         f.write('\n' + '\n' + 'args.learning_rate changed :' + str(
        #             args.learning_rate) + '\n' + '\n')  # + str(ssim) + ','
        #     f.close()
        #     break

        # model save
        if epoch == args.epochs - 1:
            torch.save(my_model.state_dict(),
                       args.model_path + '/model_epoch%d.pth' % epoch)  # save model belongs to the last epoch

        writer.close()


def val(model, epoch, val_loader, args):

    rmse_list = []
    psnr_list = []
    ergas_list = []
    sam_list = []

    for batch_id, (ref_val, pan_val, ms_val) in enumerate(val_loader):


        ms_val = F.interpolate(ms_val, scale_factor=args.scale_ratio, mode="bicubic")
        model.eval()

        # psnr = 0
        with torch.no_grad():
            '''Set mini-batch dataset'''
            ref = to_var(ref_val).detach()
            ms = to_var(ms_val).detach()
            pan = to_var(pan_val).detach()

            #out, _, _, _, _= model(pan, ms) #, _, _, _, _, _, _
            out, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(pan, ms)# , _, _, _, _, _, _, _, _, _
            #, quary_h1, key_h1, value_h1

            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()

            rmse = calc_rmse(ref, out)
            psnr = calc_psnr(ref, out)
            ergas = calc_ergas(ref, out)
            sam = calc_sam(ref, out)
            # ssim = calc_ssim(ref, out)

            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ergas_list.append(ergas)
            sam_list.append(sam)

            with open('train.txt', 'a') as f:
                f.write('epoch:' + str(epoch) + '  ,  ' + 'rmse:' + str(rmse) + '  ,  ' + 'psnr:' + str(psnr) + '  ,  '
                        + 'ergas:' + str(ergas) + '  ,  ' + 'sam:' + str(
                    sam) + '\n')  # + str(ssim) + ','
            f.close()
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

    with open('train.txt', 'a') as f:
        f.write('time:' + str(datetime.now()) + '\n')

        f.write(
            'rmse_mean:' + str(rmse_mean) + '  ,  ' + 'rmse_var:' + str(rmse_var) + '\n' + 'psnr_mean:' + str(psnr_mean)
            + '  ,  ' + 'psnr_var:' + str(psnr_var) + '\n' + 'ergas_mean:' + str(ergas_mean) + '  ,  ' + 'ergas_var:' +
            str(ergas_var) + '\n' + 'sam_mean:' + str(sam_mean) + '  ,  ' + 'sam_var:' + str(sam_var) + '\n' + '\n' +
            '\n' + '\n')
    f.close()
    return ergas_mean, rmse_mean, psnr_mean, sam_mean  # ,ssim


if __name__ == '__main__':
    main()
