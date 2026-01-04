import math

import numpy as np
import torch
import torch.nn as nn


class Conv_nxn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_act=False, use_norm=False):
        super(Conv_nxn, self).__init__()

        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        batch_norm = nn.BatchNorm2d(out_channel) if use_norm else nn.Identity()
        activation = nn.LeakyReLU(0.2, True) if use_act else nn.Identity()

        self.block = nn.Sequential()
        self.block.add_module(name="conv_{}x{}".format(kernel_size, kernel_size), module=conv)
        self.block.add_module(name="bn", module=batch_norm)
        self.block.add_module(name="act", module=activation)

    def forward(self, x):
        return self.block(x)


class ConvTranspose_nxn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, use_act=False, use_norm=False):
        super(ConvTranspose_nxn, self).__init__()
        conv_transpose = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                            padding=padding)
        batch_norm = nn.BatchNorm2d(out_channel) if use_norm else nn.Identity()
        activation = nn.LeakyReLU(0.2, True) if use_act else nn.Identity()

        self.block = nn.Sequential()
        self.block.add_module(name="conv_transpose_{}x{}".format(kernel_size, kernel_size), module=conv_transpose)
        self.block.add_module(name="bn", module=batch_norm)
        self.block.add_module(name="act", module=activation)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim=None, use_act=False, use_norm=False):
        super(ResidualBlock, self).__init__()
        hidden_dim = out_channel if hidden_dim is None else hidden_dim
        batch_norm = nn.BatchNorm2d(hidden_dim) if use_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, True) if use_act else nn.Identity()

        self.block = nn.Sequential()
        self.block.add_module(name="conv_1", module=nn.Conv2d(in_channel, hidden_dim,
                                                              kernel_size=3 if in_channel == out_channel else 1,
                                                              padding=1 if in_channel == out_channel else 0,
                                                              stride=1))
        self.block.add_module(name="bn", module=batch_norm)
        self.block.add_module(name="act", module=self.activation)
        self.block.add_module(name="conv_2",
                              module=nn.Conv2d(hidden_dim, out_channel, kernel_size=3, padding=1, stride=1))

        self.use_residual = in_channel == out_channel

    def forward(self, x):
        if self.use_residual:
            x = self.block(x) + x
        else:
            x = self.block(x)
        x = self.activation(x)
        return x


class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x


class Intra_SA(nn.Module):
    def __init__(self, dim, head_num, cross_attention=False, return_q=False):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num

        self.cross_attention = cross_attention
        self.return_q = return_q
        self.num_qkv = 2 if self.cross_attention and not self.return_q else 3

        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * self.num_qkv)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * self.num_qkv)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x, cross_q=None):
        assert (self.cross_attention and cross_q is not None) or \
               (not self.cross_attention and cross_q is None)

        h = x
        B, C, H, W = x.size()
        # assert H == W

        if H == W:
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)

            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
            feature_h = feature_h.view(B * H, W, C // 2)
            feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
            feature_v = feature_v.view(B * W, H, C // 2)
            qkv_h = torch.chunk(self.qkv_local_h(feature_h), self.num_qkv, dim=2)
            qkv_v = torch.chunk(self.qkv_local_v(feature_v), self.num_qkv, dim=2)

            if self.num_qkv == 2:
                k_h, v_h = qkv_h[0], qkv_h[1]
                k_v, v_v = qkv_v[0], qkv_v[1]
                self_query = cross_q
            else:
                q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
                q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
                self_query = torch.cat((q_h, q_v), dim=0)

            query = cross_q if self.cross_attention else self_query
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)

            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        else:
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)

            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
            feature_h = feature_h.view(B * H, W, C // 2)
            feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
            feature_v = feature_v.view(B * W, H, C // 2)
            qkv_h = torch.chunk(self.qkv_local_h(feature_h), self.num_qkv, dim=2)
            qkv_v = torch.chunk(self.qkv_local_v(feature_v), self.num_qkv, dim=2)

            if self.num_qkv == 2:
                k_h, v_h = qkv_h[0], qkv_h[1]
                k_v, v_v = qkv_v[0], qkv_v[1]
                self_query = cross_q
            else:
                q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
                q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
                self_query = q_h, q_v

            query = cross_q if self.cross_attention else self_query
            if len(self_query) == 2:
                attention_output_h = self.attn(query[0], k_h, v_h)
                attention_output_v = self.attn(query[1], k_v, v_v)
            else:
                attention_output_h = self.attn(query, k_h, v_h)
                attention_output_v = self.attn(query, k_v, v_v)

            attention_output_h = attention_output_h.view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        if self.return_q:
            return x, self_query
        return x


class Inter_SA(nn.Module):
    def __init__(self, dim, head_num, cross_attention=False, return_q=False):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num

        self.cross_attention = cross_attention
        self.return_q = return_q
        self.num_qkv = 2 if self.cross_attention and not self.return_q else 3

        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size // 2, self.num_qkv * (self.hidden_size // 2), kernel_size=1,
                                padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size // 2, self.num_qkv * (self.hidden_size // 2), kernel_size=1,
                                padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x, cross_q=None):
        assert (self.cross_attention and cross_q is not None) or \
               (not self.cross_attention and cross_q is None)

        h = x
        B, C, H, W = x.size()
        # assert H == W

        if H == W:
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)

            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = torch.chunk(self.conv_h(x_input[0]), self.num_qkv, dim=1)
            feature_v = torch.chunk(self.conv_v(x_input[1]), self.num_qkv, dim=1)

            horizontal_groups = torch.cat(feature_h, dim=0)
            horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
            horizontal_groups = horizontal_groups.view(self.num_qkv * B, H, -1)
            horizontal_groups = torch.chunk(horizontal_groups, self.num_qkv, dim=0)

            vertical_groups = torch.cat(feature_v, dim=0)
            vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
            vertical_groups = vertical_groups.view(self.num_qkv * B, W, -1)
            vertical_groups = torch.chunk(vertical_groups, self.num_qkv, dim=0)

            if self.num_qkv == 2:
                key_h, value_h = horizontal_groups[0], horizontal_groups[1]
                key_v, value_v = vertical_groups[0], vertical_groups[1]
                self_query = cross_q
            else:
                query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]
                query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]
                self_query = torch.cat((query_h, query_v), dim=0)

            query = cross_q if self.cross_attention else self_query
            key = torch.cat((key_h, value_h), dim=0)
            value = torch.cat((key_v, value_v), dim=0)

            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)

            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = torch.chunk(self.conv_h(x_input[0]), self.num_qkv, dim=1)
            feature_v = torch.chunk(self.conv_v(x_input[1]), self.num_qkv, dim=1)

            horizontal_groups = torch.cat(feature_h, dim=0)
            horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
            horizontal_groups = horizontal_groups.view(self.num_qkv * B, H, -1)
            horizontal_groups = torch.chunk(horizontal_groups, self.num_qkv, dim=0)

            vertical_groups = torch.cat(feature_v, dim=0)
            vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
            vertical_groups = vertical_groups.view(self.num_qkv * B, W, -1)
            vertical_groups = torch.chunk(vertical_groups, self.num_qkv, dim=0)

            if self.num_qkv == 2:
                key_h, value_h = horizontal_groups[0], horizontal_groups[1]
                key_v, value_v = vertical_groups[0], vertical_groups[1]
                self_query = cross_q
            else:
                query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]
                query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]
                self_query = query_h, query_v

            query = cross_q if self.cross_attention else self_query

            if len(self_query) == 2:
                attention_output_h = self.attn(query[0], key_h, value_h)
                attention_output_v = self.attn(query[1], key_v, value_v)
            else:
                attention_output_h = self.attn(query, key_h, value_h)
                attention_output_v = self.attn(query, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        if self.return_q:
            return x, self_query
        return x


class Intra_Inter_SA(nn.Module):
    def __init__(self, dim, head_num, cross_attention=False, return_q=False):
        super(Intra_Inter_SA, self).__init__()
        self.return_q = return_q
        self.intra_sa = Intra_SA(dim, head_num, cross_attention=cross_attention, return_q=return_q)
        self.inter_sa = Inter_SA(dim, head_num, cross_attention=cross_attention, return_q=return_q)

    def forward(self, x, cross_q_intra=None, cross_q_inter=None):
        if self.return_q:
            x, q_intra = self.intra_sa(x, cross_q=cross_q_intra)
            x, q_inter = self.inter_sa(x, cross_q=cross_q_inter)
            return x, q_intra, q_inter
        else:
            x = self.intra_sa(x, cross_q=cross_q_intra)
            x = self.inter_sa(x, cross_q=cross_q_inter)
            return x


class SpatialAttention(nn.Module):
    def __init__(self, mode='Baseline'):
        super(SpatialAttention, self).__init__()
        self.mode = mode
        self.main_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.sub_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()
        self.conv_map = nn.Conv2d(2, 1, 1)
        self.edge_map = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Sigmoid()
        )

        self.main_pool_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.p = nn.Sequential(
            # nn.Conv2d(4, 1, 7, padding=3),
            nn.Conv2d(4, 1, 1),
            nn.Tanh()
        )

        self.r = nn.Sequential(
            # nn.Conv2d(4, 2, 7, padding=3),
            nn.Conv2d(4, 2, 1),
            nn.Sigmoid()
        )
        self.z = nn.Sequential(
            # nn.Conv2d(4, 1, 7, padding=3),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_sub, edge_map, return_attn_map=False):
        # main & sub attention map 생성
        x_avg_pool = torch.mean(x, 1).unsqueeze(1)
        x_max_pool = torch.max(x, 1)[0].unsqueeze(1)
        x_merge_pooling = torch.cat((x_avg_pool, x_max_pool), dim=1)

        x_sub_avg_pool = torch.mean(x_sub, 1).unsqueeze(1)
        x_sub_max_pool = torch.max(x_sub, 1)[0].unsqueeze(1)
        sub_merge_pooling = torch.cat((x_sub_avg_pool, x_sub_max_pool), dim=1)

        sub_spatial_attention = self.sub_conv(sub_merge_pooling)

        if self.mode == 'Baseline':  # Baseline
            x_spatial_attention = self.main_conv(x_merge_pooling)
            if return_attn_map:
                return x * x_spatial_attention, x_sub * sub_spatial_attention, x_spatial_attention, sub_spatial_attention
            return x * x_spatial_attention, x_sub * sub_spatial_attention

        elif self.mode == 'Max':  # Max
            x_spatial_attention = self.main_conv(x_merge_pooling)
            max_attention_map = torch.maximum(x_spatial_attention, sub_spatial_attention)
            return x * max_attention_map, x_sub * sub_spatial_attention

        elif self.mode == 'Conv':  # Conv
            x_spatial_attention = self.main_conv(x_merge_pooling)
            merge_spatial_map = torch.cat((x_spatial_attention, sub_spatial_attention), dim=1)
            return x * self.conv_map(merge_spatial_map), x_sub * sub_spatial_attention

        elif self.mode == 'Edge_map':  # Edge map
            x_spatial_attention = self.main_conv(x_merge_pooling)
            edge_attention = ((sub_spatial_attention * self.edge_map(edge_map)) +
                              x_spatial_attention * (1 - self.edge_map(edge_map)))
            return x * edge_attention, x_sub * sub_spatial_attention

        elif self.mode == 'LSTM':  # LSTM-like
            merge_pool = torch.cat((x_merge_pooling, sub_merge_pooling), dim=1)
            z = self.z(merge_pool)
            r = self.r(merge_pool)
            p = self.p(torch.cat((r * x_merge_pooling, sub_merge_pooling), dim=1))
            x_spatial_attention = ((1 - z) * self.main_pool_conv(x_merge_pooling)) + (z * p)
            x_spatial_attention = self.sig(x_spatial_attention)
            if return_attn_map:
                return x * x_spatial_attention, x_sub * sub_spatial_attention, x_spatial_attention, sub_spatial_attention
            return x * x_spatial_attention, x_sub * sub_spatial_attention

        else:
            raise NotImplementedError


class Paired_Intra_Inter_CA(nn.Module):
    def __init__(self, dim, head_num, reverse=False):
        super(Paired_Intra_Inter_CA, self).__init__()
        self.reverse = reverse
        self.intra_sa = Intra_SA(dim, head_num, cross_attention=reverse, return_q=not reverse)
        self.inter_sa = Inter_SA(dim, head_num, cross_attention=reverse, return_q=not reverse)
        self.sub_intra_sa = Intra_SA(dim, head_num, cross_attention=not reverse, return_q=reverse)
        self.sub_inter_sa = Inter_SA(dim, head_num, cross_attention=not reverse, return_q=reverse)

    def forward(self, x, x_sub):
        if self.reverse:
            x_sub, q_intra = self.sub_intra_sa(x_sub)
            x = self.intra_sa(x, cross_q=q_intra)
            x_sub, q_inter = self.sub_inter_sa(x_sub)
            x = self.inter_sa(x, cross_q=q_inter)
        else:
            x, q_intra = self.intra_sa(x)
            x_sub = self.sub_intra_sa(x_sub, cross_q=q_intra)
            x, q_inter = self.inter_sa(x)
            x_sub = self.sub_inter_sa(x_sub, cross_q=q_inter)

        return x, x_sub


class Paired_Intra_Inter_Spatial(nn.Module):
    def __init__(self, dim, head_num, mode='Baseline'):
        super(Paired_Intra_Inter_Spatial, self).__init__()
        self.intra_inter = Intra_Inter_SA(dim, head_num)
        self.sub_intra_inter = Intra_Inter_SA(dim, head_num)
        self.spatial_attention = SpatialAttention(mode=mode)

    def forward(self, x, x_sub, edge_map, return_latent=False):
        x = self.intra_inter(x)
        x_sub = self.sub_intra_inter(x_sub)
        x, x_sub, latent_x, latent_x_sub = self.spatial_attention(x, x_sub, edge_map, return_attn_map=True)
        if return_latent:
            return x, x_sub, latent_x, latent_x_sub
        return x, x_sub
