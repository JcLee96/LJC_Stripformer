import torch
import torch.nn as nn

from models.blocks import Intra_Inter_SA, Conv_nxn, ConvTranspose_nxn, ResidualBlock


class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()

        self.en_layer1_1 = Conv_nxn(3, 64, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)
        self.en_layer1_2 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.en_layer1_3 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.en_layer1_4 = ResidualBlock(64, 64, use_act=True, use_norm=False)

        self.en_layer2_1 = Conv_nxn(64, 128, kernel_size=3, stride=2, padding=1, use_act=True, use_norm=False)
        self.en_layer2_2 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.en_layer2_3 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.en_layer2_4 = ResidualBlock(128, 128, use_act=True, use_norm=False)

        self.en_layer3_1 = Conv_nxn(128, 320, kernel_size=3, stride=2, padding=1, use_act=True, use_norm=False)

    def forward(self, x):
        hx = self.en_layer1_1(x)
        hx = self.en_layer1_2(hx)
        hx = self.en_layer1_3(hx)
        hx = self.en_layer1_4(hx)
        residual_1 = hx

        hx = self.en_layer2_1(hx)
        hx = self.en_layer2_2(hx)
        hx = self.en_layer2_3(hx)
        hx = self.en_layer2_4(hx)
        residual_2 = hx

        hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2


class Embeddings_output(nn.Module):
    def __init__(self):
        super(Embeddings_output, self).__init__()

        head_num = 3
        dim = 192

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = ConvTranspose_nxn(320, 192, kernel_size=4, stride=2, padding=1, use_act=True, use_norm=False)

        self.de_layer2_2 = Conv_nxn(192 + 128, 192, kernel_size=1, stride=1, padding=0, use_act=True, use_norm=False)

        self.de_block_1 = Intra_Inter_SA(dim, head_num)
        self.de_block_2 = Intra_Inter_SA(dim, head_num)
        self.de_block_3 = Intra_Inter_SA(dim, head_num)

        self.de_layer2_1 = ConvTranspose_nxn(192, 64, kernel_size=4, stride=2, padding=1, use_act=True, use_norm=False)
        self.de_layer1_3 = ResidualBlock(128, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.de_layer1_2 = ResidualBlock(64, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.de_layer1_1 = Conv_nxn(64, 3, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)

    def forward(self, x, residual_1, residual_2):
        hx = self.de_layer3_1(x)

        hx = torch.cat((hx, residual_2), dim=1)
        hx = self.de_layer2_2(hx)
        hx = self.de_block_1(hx)
        hx = self.de_block_2(hx)
        hx = self.de_block_3(hx)
        hx = self.de_layer2_1(hx)

        hx = torch.cat((hx, residual_1), dim=1)
        hx = self.de_layer1_3(hx)
        hx = self.de_layer1_2(hx)
        hx = self.de_layer1_1(hx)

        return hx


class Stripformer(nn.Module):
    def __init__(self, dim=320, head_num=5):
        super(Stripformer, self).__init__()
        self.encoder = Embeddings()
        self.Trans_block_1 = Intra_Inter_SA(dim, head_num)
        self.Trans_block_2 = Intra_Inter_SA(dim, head_num)
        self.Trans_block_3 = Intra_Inter_SA(dim, head_num)
        self.Trans_block_4 = Intra_Inter_SA(dim, head_num)
        self.Trans_block_5 = Intra_Inter_SA(dim, head_num)
        self.Trans_block_6 = Intra_Inter_SA(dim, head_num)
        self.decoder = Embeddings_output()

    def forward(self, x):
        hx, residual_1, residual_2 = self.encoder(x)
        hx = self.Trans_block_1(hx)
        hx = self.Trans_block_2(hx)
        hx = self.Trans_block_3(hx)

        hx = self.Trans_block_4(hx)
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)
        hx = self.decoder(hx, residual_1, residual_2)
        return hx + x
