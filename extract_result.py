import os.path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as SSIM

from util.metrics import PSNR


def get_file_paths(base_dir):
    file_paths = {}
    for path, dir, files in os.walk(base_dir):
        for filename in files:
            ext = filename.split('.')[-1]
            if ext.lower() != "png":
                continue
            if filename in file_paths:
                raise AssertionError
            file_paths[filename] = os.path.join(path, filename)
    return file_paths


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
    return image_numpy


if __name__ == '__main__':
    csv_path = "./iter_200_result.csv"

    gt_dir = "/ssd1/LJC/New_BDD_100k/frames/100k/test"
    blur_dir = "/data1/LJC/New_BDD_100k/frames_avg/100k/test"
    pred_1_dir = "/home/compu/LJC/Stripformer/out/Stripformer_bbd100k_results_v"
    pred_2_dir = "/data1/LJC/hi_diff/results/test_HI_Diff_GoPro/visualization/ValSet"

    save_dir = "./temp_visualize_SSIM_diffusion_compare"
    os.makedirs(save_dir, exist_ok=True)

    gt_paths = get_file_paths(gt_dir)
    print(len(gt_paths))
    blur_paths = get_file_paths(blur_dir)
    print(len(blur_paths))
    pred_1_paths = get_file_paths(pred_1_dir)
    print(len(pred_1_paths))
    pred_2_paths = get_file_paths(pred_2_dir)
    print(len(pred_2_paths))

    with open(csv_path, "r") as rf:
        for line in rf.readlines():
            line_split = line.strip("\n").split(",")

            blur_path = line_split[0]

            # psnr = float(line_split[1])
            # ssim = float(line_split[2])
            #
            # psnr_blur = float(line_split[3])
            # ssim_blur = float(line_split[4])

            filename = os.path.basename(blur_path)
            # sub_filename = "predict_" + filename.replace("100.png", "102.png")

            if filename not in gt_paths:
                print("GT", filename)
                raise AssertionError
            elif filename not in pred_1_paths:
                print("pred_1", filename)
                raise AssertionError
            elif filename not in pred_2_paths:
                print("pred_2", filename)
                raise AssertionError

            gt_path = gt_paths[filename]
            pred_1_path = pred_1_paths[filename]
            pred_2_path = pred_2_paths[filename]

            blur = cv2.imread(blur_path)
            gt = cv2.imread(gt_path)
            pred_1 = cv2.imread(pred_1_path)
            pred_2 = cv2.imread(pred_2_path)

            psnr_blur = PSNR(blur, gt)
            ssim_blur = SSIM(blur, gt, multichannel=True)

            psnr_vanilla = PSNR(pred_1, gt)
            ssim_vanilla = SSIM(pred_1, gt, multichannel=True)

            psnr = PSNR(pred_2, gt)
            ssim = SSIM(pred_2, gt, multichannel=True)

            # if ssim < ssim_vanilla:
            #     continue

            # pred_2 = cv2.resize(pred_2, (pred_2.shape[1] * 2, pred_2.shape[0] * 2))

            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            blur = cv2.rotate(blur, cv2.ROTATE_90_CLOCKWISE)
            pred_1 = cv2.rotate(pred_1, cv2.ROTATE_90_CLOCKWISE)
            pred_2 = cv2.rotate(pred_2, cv2.ROTATE_90_CLOCKWISE)
            stacked = np.hstack((gt, blur, pred_1, pred_2))

            gt = cv2.rotate(gt, cv2.ROTATE_180)
            blur = cv2.rotate(blur, cv2.ROTATE_180)
            pred_1 = cv2.rotate(pred_1, cv2.ROTATE_180)
            pred_2 = cv2.rotate(pred_2, cv2.ROTATE_180)
            stacked = np.vstack((stacked, np.hstack((gt, blur, pred_1, pred_2))))

            txt_line = "PSNR B: {}    SSIM B: {}    PSNR V: {}    SSIM V: {}    PSNR S: {}    SSIM S: {}".format(
                round(psnr_blur, 2), round(ssim_blur, 4), round(psnr_vanilla, 2), round(ssim_vanilla, 4), round(psnr, 2), round(ssim, 4)
            )
            if (ssim - ssim_blur) / ssim_blur < 0.01:
                print("SSIM", filename, (ssim - ssim_blur) / ssim_blur)
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            stacked = cv2.putText(stacked, txt_line, (60, 60), cv2.FONT_HERSHEY_PLAIN, 3, color, 2, cv2.LINE_AA)

            save_path = os.path.join(save_dir, filename)
            print(save_path)
            cv2.imwrite(save_path, stacked)
