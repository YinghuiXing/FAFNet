import torch
import numpy as np

from PIL import Image
from scipy.signal import convolve2d

'''
psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print('RMSE:   {:.4f};'.format(rmse))
    print('PSNR:   {:.4f};'.format(psnr))
    print('ERGAS:   {:.4f};'.format(ergas))
    print('SAM:   {:.4f}.'.format(sam))

'''


def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)  # -1 means it can be calculated by the other information
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt - img_fus) ** 2, axis=1)
    rmse = rmse ** 0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse / mean) ** 2)
    ergas = 100 / 4 * ergas ** 0.5  # yi wen  band=4 can stand but band=8 can?

    return ergas


def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt - img_fus) ** 2)
    img_max = np.max(img_tgt)
    psnr = 10 * np.log10(img_max ** 2 / mse)

    return psnr


def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt - img_fus) ** 2))

    return rmse


def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    # calculate
    # img_tgt**2 element wise, Square of number
    # np.sum(img_tgt**2, axis=0) add elements that belong to the same column,return a row of vector
    A = np.sqrt(np.sum(img_tgt ** 2, axis=0))

    # np.sqrt  sqrt every element
    B = np.sqrt(np.sum(img_fus ** 2, axis=0))
    AB = np.sum(img_tgt * img_fus, axis=0)

    # img_tgt*img_fus dot product
    sam = AB / (A * B + 0.0000000001)  # if A*B==0  it will return nan,then nan will pass down
    # print('sam   A：{}    B:{}     AB:{}    A * B:{}   '.format(A, B, AB, A * B))
    # print('A * B:{}   AB / (A * B):{}'.format(A * B, sam))
    sam = np.arccos(sam)
    # print('np.arccos(sam):{}'.format(sam))
    sam = np.mean(sam) * 180 / 3.1415926535
    # print('sam_final:{}'.format(sam))
    return sam
'''
sam   A：[0.60638666 0.6183532  0.5929225  ... 0.6920875  0.83845484 0.93456835]    B:[0.2020485  0.37715033 0.30675852 ... 0.49520892 0.45900744 0.52376926]     AB:[0.10607854 0.2295825  0.17566234 ... 0.33001947 0.36427113 0.4791539 ]    A * B:[0.12251952 0.23321211 0.18188404 ... 0.3427279  0.384857   0.48949817]   
A * B:[0.12251952 0.23321211 0.18188404 ... 0.3427279  0.384857   0.48949817]   AB / (A * B):[0.8658093  0.98443645 0.9657931  ... 0.9629198  0.9465104  0.9788676 ]
np.arccos(sam):[0.5240308  0.17665835 0.26231182 ... 0.27317277 0.3285526  0.20594786]
sam_final:nan

'''

#
def calc_cc(img_pan, img_ms):
    img_pan = np.squeeze(img_pan)
    img_ms = np.squeeze(img_ms)
    img_pan = img_pan.reshape(img_pan.shape[0], -1)
    img_ms = img_ms.reshape(img_ms.shape[0], -1)
    img_pan = img_pan / np.max(img_pan)
    img_ms = img_ms / np.max(img_ms)

    mean_pan = np.mean(img_pan)
    mean_ms = np.mean(img_ms)

    pan_qm = img_pan - mean_pan
    ms_qm = img_ms - mean_ms

    A = np.sum(pan_qm * ms_qm)
    B = np.sqrt(np.sum((pan_qm ** 2)) * np.sum((ms_qm ** 2)))
    cc = A / B
    return cc


# input: [n,c,h,w] tensor     output: a number
def calc_cc_tensor(img_pan, img_ms):
    img_pan = img_pan.squeeze()
    img_ms = img_ms.squeeze()
    img_pan = img_pan.reshape(img_pan.shape[0], -1)
    img_ms = img_ms.reshape(img_ms.shape[0], -1)
    img_pan = img_pan / torch.max(img_pan)
    img_ms = img_ms / torch.max(img_ms)

    pan_qm = img_pan - torch.mean(img_pan)
    ms_qm = img_ms - torch.mean(img_ms)

    A = torch.sum(pan_qm * ms_qm)
    B = (torch.sum((pan_qm ** 2)) * torch.sum((ms_qm ** 2))).sqrt()
    cc = A / B
    return cc


# def calc_ssim(img_tgt, img_fus):
#
#
#
#     return ssim


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calc_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    im1 = np.squeeze(im1).reshape(im1.shape[0], -1)
    im2 = np.squeeze(im2).reshape(im2.shape[0], -1)

    # M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


if __name__ == "__main__":
    img_path = ('1.jpeg')
    img = Image.open(img_path)
    img_arr = np.array(img)

    print(calc_ssim(img_arr[:, :, 2], img_arr[:, :, 1]))
