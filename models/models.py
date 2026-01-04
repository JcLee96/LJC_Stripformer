import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM

from util.metrics import PSNR
import torch

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        sub = data['c']
        sub_targets = data['d']
        edge_sub = data['e']

        inputs, targets, sub, sub_targets, edge_sub = (inputs.cuda(), targets.cuda(), sub.cuda(), sub_targets.cuda(), edge_sub.cuda())
        return inputs, targets, torch.cat((sub, edge_sub), dim=1), sub_targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        return image_numpy

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
