from __future__ import print_function

import argparse
import os
import re
import time

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision
import yaml
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable

from models.networks import get_generator
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


def threshold_image(image_array):
    # 랜덤한 threshold 값 생성
    threshold_value = np.random.randint(0.1 * 255., 0.3 * 255.)

    # 이미지의 각 픽셀에 랜덤한 threshold 값을 적용하고, threshold 값보다 작은 부분을 0으로 설정
    thresholded_array = np.where(image_array < threshold_value, 0, image_array)
    return thresholded_array.astype(np.uint8)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    args = get_args()
    with open('config/config_Stripformer_gopro3.yaml') as cfg:
        config = yaml.safe_load(cfg)

    blur_path = '/data1/LJC/New_BDD_100k/frames_avg/100k/test'
    out_path = './out/Stripformer_bbd100k_results_Max2'
    sub_out_path = './out/Stripformer_bbd100k_results_Max_subimg2'

    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if not os.path.isdir(sub_out_path):
        os.mkdir(sub_out_path)

    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights_path))
    model = model.cuda()

    test_time = 0
    iteration = 0
    total_image_number = 20001
    sub_resize = iaa.Resize({"height": 720 // 2, "width": 1280 // 2})


    def tensor2im(image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        return image_numpy


    def calculate_sigma_color(image, k=0.01):
        # 각 채널에 대해 색상 차이 계산
        color_diff = np.abs(image - np.mean(image, axis=(0, 1)))

        # 각 채널에 대한 색상 차이에 대한 표준편차 계산
        sigma_color = np.std(color_diff, axis=(0, 1))

        # 각 픽셀의 색상 차이의 평균을 계산하여 전체 채널에 대한 sigma_color로 사용
        sigma_color = np.mean(k * sigma_color)

        return sigma_color


    def radius_filter(image):
        # FFT 수행
        spectrum = fftshift(fft2(image))
        image_size_x, image_size_y = image.shape[0], image.shape[1]

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


    def make_subframe(frame, target_frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        # resize
        frame = sub_resize.augment_image(frame)
        target_frame = sub_resize.augment_image(target_frame)

        # thresholding
        threshold_value = np.random.randint(0.1 * 255., 0.3 * 255.)

        if np.max(frame) <= threshold_value:
            th_frame = frame
        else:
            th_frame = np.where(frame < threshold_value, 0, frame).astype(np.uint8)

        if np.max(target_frame) <= threshold_value:
            th_target_frame = target_frame
        else:
            th_target_frame = np.where(target_frame < threshold_value, 0, target_frame).astype(np.uint8)

        th_frame = (th_frame - np.min(th_frame)) / (np.max(th_frame) - np.min(th_frame)) * 255
        th_target_frame = (th_target_frame - np.min(th_target_frame)) / (np.max(th_target_frame) - np.min(th_target_frame)) * 255

        th_frame = th_frame.astype(np.uint8)
        th_target_frame = th_target_frame.astype(np.uint8)

        # gaussian noise
        gaussian_noise = np.random.uniform(0.05, 0.15)
        gaussian_noise_aug = iaa.AdditiveGaussianNoise(scale=gaussian_noise * 255)
        gau_th_frame = gaussian_noise_aug(images=th_frame)

        sigmaColor = calculate_sigma_color(image=gau_th_frame, k=0.01)
        gau_th_frame = cv2.GaussianBlur(gau_th_frame, (5, 5), sigmaColor)

        edge_th_frame = radius_filter(gau_th_frame)

        return gau_th_frame, th_target_frame, edge_th_frame, th_frame

    csv_path = "./Max_Stripformer.csv"
    import csv

    results_list = []

    with open(csv_path, mode='w', newline='') as wf:
        writer = csv.writer(wf)

        # 결과 CSV 파일에 헤더 행 쓰기
        writer.writerow(['path', 'psnr', 'ssim', 'min', 'max', 'avg', 'std'])

        for file in os.listdir(blur_path):
            if not os.path.isdir(out_path + '/' + file):
                os.mkdir(out_path + '/' + file)
            if not os.path.isdir(sub_out_path + '/' + file):
                os.mkdir(sub_out_path + '/' + file)

            for img_name in os.listdir(blur_path + '/' + file):
                img = cv2.imread(blur_path + '/' + file + '/' + img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                sub_img_path = (add_two_to_number(blur_path + '/' + file + '/' + img_name).replace("frames_avg", "frames").replace("/data1", "/ssd1"))
                origin_sub_img_path = ((blur_path + '/' + file + '/' + img_name).replace("frames_avg", "frames").replace("/data1", "/ssd1"))

                sub_img = cv2.imread(sub_img_path)
                ori_sub_img = cv2.imread(origin_sub_img_path)
                target_blur_img = ori_sub_img
                sub_img, origin_img, edge_img, th_sub_img = make_subframe(sub_img, ori_sub_img)

                img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32'))
                target_blur_img = torch.from_numpy(np.transpose(target_blur_img / 255, (2, 0, 1)).astype('float32'))
                sub_img = np.expand_dims(sub_img, axis=0)
                sub_img = torch.from_numpy((sub_img / 255).astype('float32'))
                origin_img = np.expand_dims(origin_img, axis=0)
                origin_img = torch.from_numpy((origin_img / 255).astype('float32'))
                edge_img = np.expand_dims(edge_img, axis=0)
                edge_img = torch.from_numpy((edge_img / 255).astype('float32'))

                th_sub_img = np.expand_dims(th_sub_img, axis=0)
                th_sub_img = torch.from_numpy((th_sub_img / 255).astype('float32'))

                new_sub_img = torch.cat((sub_img, edge_img), dim=0)
                with torch.no_grad():
                    iteration += 1
                    img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
                    target_blur_img = Variable(target_blur_img.unsqueeze(0)).cuda()
                    sub_img = Variable(sub_img.unsqueeze(0)).cuda()
                    new_sub_img = Variable(new_sub_img.unsqueeze(0)).cuda()

                    th_sub_img = Variable(th_sub_img.unsqueeze(0)).cuda()
                    edge_img = Variable(edge_img.unsqueeze(0)).cuda()

                    start = time.time()

                    result_image, sub_image = model(img_tensor, new_sub_img, return_subframe=True)
                    stop = time.time()
                    print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
                    test_time += stop - start
                    print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))

                    inp = tensor2im(img_tensor)
                    fake = tensor2im(result_image.data)
                    real = tensor2im(target_blur_img.data)

                    psnr = PSNR(fake, real)
                    ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), multichannel=True)

                    min = np.min(real)
                    max = np.max(real)
                    avg = np.average(real)
                    std = np.std(real)

                    writer.writerow([img_name, psnr, ssim, min, max, avg, std])

