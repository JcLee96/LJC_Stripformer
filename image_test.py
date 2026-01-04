from __future__ import print_function

import argparse
import os
import re

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable

from util.metrics import PSNR


def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	return parser.parse_args()


def add_two_to_number(file_path):
    match = re.search(r'(\d+)(?=\.\w+$)', file_path)
    if match:
        number = int(match.group())
        new_number = number + 2
        return re.sub(r'(\d+)(?=\.\w+$)', str(new_number), file_path)
    else:
        return file_path

def calculate_sigma_color(rgb_image, k):
    # 각 채널에 대해 색상 차이 계산

    color_diff = np.abs(rgb_image - np.mean(rgb_image, axis=(0, 1)))

    # 각 채널에 대한 색상 차이에 대한 표준편차 계산
    sigma_color = np.std(color_diff, axis=(0, 1))

    # 각 픽셀의 색상 차이의 평균을 계산하여 전체 채널에 대한 sigma_color로 사용
    sigma_color = np.mean(k*sigma_color)

    return sigma_color

def radius_filter(image):
    # FFT 수행
    spectrum = fftshift(fft2(image))

    image_size_x, image_size_y = image.shape[0], image.shape[1]

    # 원의 반지름 설정

    # 원의 중앙 좌표 설정
    center_x, center_y = image_size_x // 2, image_size_y // 2

    # 원의 반지름 설정
    radius = 30  # 각 구역의 지름은 20이므로 반지름은 10

    y, x = np.ogrid[:image_size_x, :image_size_y]
    # 오른쪽 상단
    mask_top_right = (x - (image_size_y - center_y)) ** 2 + (y - center_x) ** 2 <= radius ** 2
    spectrum[mask_top_right] = 0


    # IFFT 수행
    filtered_image = np.abs(ifft2(ifftshift(spectrum)))

    return filtered_image

if __name__ == '__main__':
    blur_path = '/data1/LJC/New_BDD_100k/frames_avg/100k/train'

    test_time = 0
    iteration = 0
    total_bi_psnr, total_bi_ssim, total_psnr = 0, 0, 0
    total_ga_psnr, total_ga_ssim, total_ssim = 0, 0, 0

    total_image_number = 20001
    sub_resize = iaa.Resize({"height": 1280 // 2, "width": 720 // 2})

    def tensor2im(image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
        return image_numpy

    def make_subframe(frame):
        frame = cv2.imread(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # resize
        frame = sub_resize.augment_image(frame)

        # thresholding
        frame = frame.astype(int)

        # thresholding
        threshold_value = np.random.randint(int(0.1 * np.min(frame)), int(0.3 * np.max(frame))+1)
        frame = np.where(frame < threshold_value, 0, frame).astype(np.uint8)

        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
        frame = frame.astype(np.uint8)

        # gaussian noise
        gaussian_noise = np.random.uniform(0.05, 0.15)
        gaussian_noise_aug = iaa.AdditiveGaussianNoise(scale=gaussian_noise * 255)
        gau_frame = gaussian_noise_aug(images=frame)

        return gau_frame, frame


    for file in os.listdir(blur_path):
        for img_name in os.listdir(blur_path + '/' + file):
            iteration += 1
            data_path = ''
            sub_img_path = ((blur_path + '/' + file + '/' + img_name).replace("frames_avg", "frames")
                            .replace("/data1", "/ssd1"))

            sub_img, gt_sub_img = make_subframe(sub_img_path)


            sigmaColor = calculate_sigma_color(sub_img, 1)
            bi_sub_img = cv2.bilateralFilter(sub_img, -1, sigmaColor, 5)

            ga_sub_img = cv2.GaussianBlur(sub_img, (5, 5), sigmaColor)

            # bi_filter_sub_img = radius_filter(bi_sub_img)
            # gau_filter_sub_img = radius_filter(ga_sub_img)


            sub_img = np.expand_dims(sub_img, axis=0)
            sub_img = torch.from_numpy((sub_img / 255).astype('float32'))

            gt_sub_img = np.expand_dims(gt_sub_img, axis=0)
            gt_sub_img = torch.from_numpy((gt_sub_img / 255).astype('float32'))

            # bi_sub_img = np.expand_dims(bi_sub_img, axis=0)
            # bi_sub_img = torch.from_numpy((bi_sub_img / 255).astype('float32'))
            #
            # ga_sub_img = np.expand_dims(ga_sub_img, axis=0)
            # ga_sub_img = torch.from_numpy((ga_sub_img / 255).astype('float32'))

            # bi_filter_sub_img = np.expand_dims(bi_filter_sub_img, axis=0)
            # bi_filter_sub_img = torch.from_numpy((bi_filter_sub_img / 255).astype('float32'))
            #
            # gau_filter_sub_img = np.expand_dims(gau_filter_sub_img, axis=0)
            # gau_filter_sub_img = torch.from_numpy((gau_filter_sub_img / 255).astype('float32'))

            sub_img = Variable(sub_img.unsqueeze(0)).cuda()
            gt_sub_img = Variable(gt_sub_img.unsqueeze(0)).cuda()

            # bi_sub_img = Variable(bi_sub_img.unsqueeze(0)).cuda()
            # ga_sub_img = Variable(ga_sub_img.unsqueeze(0)).cuda()
            #
            # bi_filter_sub_img = Variable(bi_filter_sub_img.unsqueeze(0)).cuda()
            # gau_filter_sub_img = Variable(gau_filter_sub_img.unsqueeze(0)).cuda()

            # fake = tensor2im(sub_img.data)

            # bi_filter_fake = tensor2im(bi_sub_img.data)
            # ga_filter_fake = tensor2im(ga_sub_img.data)
            #
            # bi_filter_sub_img_fake = tensor2im(bi_filter_sub_img.data)
            # gau_filter_sub_img_fake = tensor2im(gau_filter_sub_img.data)

            # real = tensor2im(gt_sub_img.data)
            #
            # psnr = PSNR(fake, real)
            # ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), multichannel=True)
            #
            # total_psnr += psnr
            # total_ssim += ssim

            # bi_psnr = PSNR(bi_filter_sub_img_fake, real)
            # bi_ssim = SSIM(bi_filter_sub_img_fake.astype('uint8'), real.astype('uint8'), multichannel=True)
            #
            # ga_psnr = PSNR(gau_filter_sub_img_fake, real)
            # ga_ssim = SSIM(gau_filter_sub_img_fake.astype('uint8'), real.astype('uint8'), multichannel=True)
            #
            # total_bi_psnr += bi_psnr
            # total_bi_ssim += bi_ssim
            #
            # total_ga_psnr += ga_psnr
            # total_ga_ssim += ga_ssim
    # print("bi_psnr_score", total_psnr / iteration)
    # print("bi_ssim_score", total_ssim / iteration)

    # print("bi_psnr_score", total_bi_psnr / iteration)
    # print("bi_ssim_score", total_bi_ssim / iteration)
    #
    # print("ga_psnr_score", total_ga_psnr / iteration)
    # print("ga_ssim_score", total_ga_ssim / iteration)

            # print("psnr_score", psnr)
            # print("ssim_score", ssim)

