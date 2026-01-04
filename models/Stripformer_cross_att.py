import torch
import torch.nn as nn

from .blocks import Conv_nxn, ConvTranspose_nxn, ResidualBlock, Paired_Intra_Inter_Spatial
# from blocks import Conv_nxn, ConvTranspose_nxn, ResidualBlock, Paired_Intra_Inter_Spatial


class PairedEncoder(nn.Module):
    def __init__(self, main_channels, sub_channels):
        super(PairedEncoder, self).__init__()

        # Encoder Main
        self.en_layer1_1 = Conv_nxn(main_channels, 64, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)
        self.en_layer1_2 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.en_layer1_3 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.en_layer1_4 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.en_layer2_1 = Conv_nxn(64, 128, kernel_size=3, stride=2, padding=1, use_act=True, use_norm=False)
        self.en_layer2_2 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.en_layer2_3 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.en_layer2_4 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.en_layer3_1 = Conv_nxn(128, 320, kernel_size=3, stride=2, padding=1, use_act=True, use_norm=False)

        # Encoder Sub
        self.sub_en_layer1_1 = Conv_nxn(sub_channels, 64, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)
        self.sub_en_layer1_2 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.sub_en_layer1_3 = ResidualBlock(64, 64, use_act=True, use_norm=False)
        self.sub_en_layer1_4 = ResidualBlock(64, 64, use_act=True, use_norm=False)

        self.sub_en_layer2_1 = Conv_nxn(64, 128, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)
        self.sub_en_layer2_2 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.sub_en_layer2_3 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.sub_en_layer2_4 = ResidualBlock(128, 128, use_act=True, use_norm=False)
        self.sub_en_layer3_1 = Conv_nxn(128, 320, kernel_size=3, stride=2, padding=1, use_act=True, use_norm=False)

    def forward(self, x, x_sub):


        x = self.en_layer1_1(x)
        x_sub = self.sub_en_layer1_1(x_sub)
        x = self.en_layer1_2(x)
        x_sub = self.sub_en_layer1_2(x_sub)
        x = self.en_layer1_3(x)
        x_sub = self.sub_en_layer1_3(x_sub)
        x = self.en_layer1_4(x)
        x_sub = self.sub_en_layer1_4(x_sub)
        res_1, res_sub_1 = x, x_sub

        x = self.en_layer2_1(x)
        x_sub = self.sub_en_layer2_1(x_sub)
        x = self.en_layer2_2(x)
        x_sub = self.sub_en_layer2_2(x_sub)
        x = self.en_layer2_3(x)
        x_sub = self.sub_en_layer2_3(x_sub)
        x = self.en_layer2_4(x)
        x_sub = self.sub_en_layer2_4(x_sub)

        res_2, res_sub_2 = x, x_sub

        x = self.en_layer3_1(x)
        x_sub = self.sub_en_layer3_1(x_sub)

        return x, res_1, res_2, x_sub, res_sub_1, res_sub_2


class PairedDecoder(nn.Module):
    def __init__(self, main_channels, sub_channels, decoder_dim=192, decoder_head_num=3):
        super(PairedDecoder, self).__init__()

        # Decoder Paired Trans
        self.paired_de_trans_block_1 = Paired_Intra_Inter_Spatial(decoder_dim, decoder_head_num)
        self.paired_de_trans_block_2 = Paired_Intra_Inter_Spatial(decoder_dim, decoder_head_num)
        self.paired_de_trans_block_3 = Paired_Intra_Inter_Spatial(decoder_dim, decoder_head_num)

        # Decoder Main
        self.de_layer3_1 = ConvTranspose_nxn(320, 192, kernel_size=4, stride=2, padding=1, use_act=True, use_norm=False)
        self.de_layer2_2 = Conv_nxn(192 + 128, 192, kernel_size=1, stride=1, padding=0, use_act=True, use_norm=False)
        # here self.paired_de_trans_blocks
        self.de_layer2_1 = ConvTranspose_nxn(192, 64, kernel_size=4, stride=2, padding=1, use_act=True, use_norm=False)
        self.de_layer1_3 = ResidualBlock(128, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.de_layer1_2 = ResidualBlock(64, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.de_layer1_1 = Conv_nxn(64, main_channels, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)

        # Decoder Sub
        self.sub_de_layer3_1 = ConvTranspose_nxn(320, 192, kernel_size=4, stride=2, padding=1, use_act=True, use_norm=False)
        self.sub_de_layer2_2 = Conv_nxn(192 + 128, 192, kernel_size=1, stride=1, padding=0, use_act=True, use_norm=False)
        # here self.paired_de_trans_blocks
        # we don't use below lines during inference
        self.sub_de_layer2_1 = Conv_nxn(192, 64, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)
        self.sub_de_layer1_3 = ResidualBlock(128, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.sub_de_layer1_2 = ResidualBlock(64, 64, hidden_dim=64, use_act=True, use_norm=False)
        self.sub_de_layer1_1 = Conv_nxn(64, 1, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=False)

    def forward(self, x, res_1, res_2, x_sub, res_sub_1, res_sub_2, edge_x, return_subframe=True):
        x = self.de_layer3_1(x)
        x = torch.cat((x, res_2), dim=1)
        x = self.de_layer2_2(x)

        x_sub = self.sub_de_layer3_1(x_sub)
        x_sub = torch.cat((x_sub, res_sub_2), dim=1)
        x_sub = self.sub_de_layer2_2(x_sub)

        x, x_sub = self.paired_de_trans_block_1(x, x_sub, edge_x)
        x, x_sub = self.paired_de_trans_block_2(x, x_sub, edge_x)
        x, x_sub = self.paired_de_trans_block_3(x, x_sub, edge_x)

        x = self.de_layer2_1(x)
        x = torch.cat((x, res_1), dim=1)
        x = self.de_layer1_3(x)
        x = self.de_layer1_2(x)
        x = self.de_layer1_1(x)

        if return_subframe == False:
            return x
        else:
            x_sub = self.sub_de_layer2_1(x_sub)
            x_sub = torch.cat((x_sub, res_sub_1), dim=1)
            x_sub = self.sub_de_layer1_3(x_sub)
            x_sub = self.sub_de_layer1_2(x_sub)
            x_sub = self.sub_de_layer1_1(x_sub)
            return x, x_sub


class StripformerMultiInputV2(nn.Module):
    def __init__(self, main_channels=3, sub_channels=2, mode=0):
        super(StripformerMultiInputV2, self).__init__()
        trans_dim = 320
        trans_head_num = 5
        decoder_dim = 192
        decoder_head_num = 3

        self.paired_encoder = PairedEncoder(main_channels, sub_channels)

        # Trans
        self.paired_trans_block_1 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)
        self.paired_trans_block_2 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)
        self.paired_trans_block_3 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)
        self.paired_trans_block_4 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)
        self.paired_trans_block_5 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)
        self.paired_trans_block_6 = Paired_Intra_Inter_Spatial(trans_dim, trans_head_num, mode)

        # Decoder Main
        self.paired_decoder = PairedDecoder(main_channels, sub_channels, decoder_dim=decoder_dim, decoder_head_num=decoder_head_num)

    def forward(self, x, x_sub, return_subframe=True):
        r, r_sub = x, x_sub
        edge_x = x_sub[:, 1:, :, :]
        # Encoder
        x, res_1, res_2, x_sub, res_sub_1, res_sub_2 = self.paired_encoder(x, x_sub)

        # Trans
        x, x_sub = self.paired_trans_block_1(x, x_sub, edge_x)
        x, x_sub = self.paired_trans_block_2(x, x_sub, edge_x)
        x, x_sub = self.paired_trans_block_3(x, x_sub, edge_x)
        x, x_sub = self.paired_trans_block_4(x, x_sub, edge_x)
        x, x_sub = self.paired_trans_block_5(x, x_sub, edge_x)
        x, x_sub = self.paired_trans_block_6(x, x_sub, edge_x)

        # Decoder
        if return_subframe == True:
            x, x_sub = self.paired_decoder(x, res_1, res_2, x_sub, res_sub_1, res_sub_2, edge_x, return_subframe)
            return x + r, x_sub + r_sub
        else:
            x = self.paired_decoder(x, res_1, res_2, x_sub, res_sub_1, res_sub_2, edge_x, return_subframe)
            return x + r


if __name__ == "__main__":
    model = StripformerMultiInputV2(main_channels=3, sub_channels=2, mode=4)

    dummy_main = torch.randn((1, 3, 512, 512))
    dummy_sub = torch.randn((1, 2, 256, 256))

    pred_main, pred_sub = model(dummy_main, dummy_sub)
    print(pred_main.shape)
    print(pred_sub.shape)
