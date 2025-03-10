from glob import glob
import numpy as np
import torch
import argparse
import lpips

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def main():
    parser = argparse.ArgumentParser(description="Argument Parsing Example")

    # 인자 추가
    parser.add_argument("--sample_path", type=str, help="SAMPLE_PATH")
    args = parser.parse_args()


    lpips_fn = lpips.LPIPS(net='alex')
    fake_files = glob(f"{args.sample_path}/*fake*.npz")
    real_files = glob(f"{args.sample_path}/*real*.npz")

    length = len(real_files)
    assert len(fake_files) == len(real_files)
    fake_files.sort()
    real_files.sort()

    MSE = 0.
    SSIM = 0.
    LPIPS = 0.
    count = 0

    for i in range(length):
        fake_ = np.load(fake_files[i])["samples"] / 255.
        real_ = np.load(real_files[i])["samples"] / 255.

        count += fake_.shape[0]

        minus = (real_ - fake_) ** 2
        mean = np.sum(minus) / (128 * 128 * 3)
        MSE += mean

        SSIM += torch.sum(ssim(torch.tensor(real_).to(torch.float64), torch.tensor(fake_), size_average=False)).item()

        i1 = torch.tensor(real_).to(torch.float32) * 2. - 1.
        i2 = torch.tensor(fake_).to(torch.float32) * 2. - 1.
        LPIPS += torch.sum(lpips_fn.forward(i1.permute(0, 3, 1, 2), i2.permute(0, 3, 1, 2))).item()

        print("For sample number:", count)
        print("MSE: {:.4f}".format(MSE / count))
        print("SSIM: {:.4f}".format(SSIM / count))
        print("LPIPS: {:.4f}".format(LPIPS / count))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type(
        ) == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel,
                     self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    main()
