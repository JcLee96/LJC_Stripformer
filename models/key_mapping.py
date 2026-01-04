import torch
import torch.nn as nn
from models.Stripformer import Stripformer
from models.Stripformer_gru_t import StripformerMultiInputV2

import pdb

# for first method
def auto_map_module_names(pretrained_dict, new_model_dict_name, new_model):
    new_model_dict = new_model.state_dict()
    mapping = {}

    for (pre_trained_name, pre_trained_param), (new_trained_name, new_trained_param), in zip(pretrained_dict.items(), new_model_dict_name.items()):
        # Pre 모듈 이름에서 중요 정보 추출
        pre_parts = pre_trained_name.split('.')
        pre_module_type = pre_parts[1]
        # pre_module_name, pre_rest_of_name = pre_parts[1], '.'.join(pre_parts[2:])

        # New 모듈 이름에서 중요 정보 추출
        new_parts = new_trained_name.split('.')
        new_module_type = new_parts[1]
        new_module_name, new_rest_of_name = new_parts[2], '.'.join(new_parts[3:])

        # 모듈 이름의 변환 규칙을 적용하여 새로운 모듈 이름 생성
        # encoder & decoder
        if 'en_layer' in new_module_name and 'sub_en_layer' not in new_module_name:
            # en_layer
            new_model_dict = new_model.state_dict()
            new_model_dict[new_trained_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)
            # sub_en_layer
            if 'en_layer1' not in new_module_name:
                if '2_1' in new_module_name or '1_1' in new_module_name:
                    pass
                # elif '2_2' in new_module_name:
                #     pass
                else:
                    new_model_dict = new_model.state_dict()
                    new_model_dict[new_trained_name.replace('en_layer', 'sub_en_layer')] = pre_trained_param
                    new_model.load_state_dict(new_model_dict)

        elif 'de_layer' in new_module_name and 'sub_de_layer' not in new_module_name:
            new_model_dict = new_model.state_dict()
            new_model_dict[new_trained_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

        # intra & inter encoder
        elif 'intra' in new_parts[3] and 'sub_intra_inter' not in new_module_name and pre_parts[1] != 'decoder' and int(pre_parts[1].split('_')[-1]) % 2 == 1:
            new_model_dict = new_model.state_dict()

            # intra
            new_model_dict['.'.join(new_parts[:4]) + '.' + '.'.join(pre_parts[2:])] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

            # sub_intra
            new_name = '.'.join(new_parts[:4]) + '.' + '.'.join(pre_parts[2:])
            new_name = new_name.replace('intra_inter', 'sub_intra_inter')
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

        elif 'inter' in new_parts[3] and 'sub_intra_inter' not in new_module_name and pre_parts[1] != 'decoder' and int(pre_parts[1].split('_')[-1]) % 2 == 0:
            new_model_dict = new_model.state_dict()


            # inter
            new_model_dict['.'.join(new_parts[:4]) + '.' + '.'.join(pre_parts[2:])] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

            # sub_inter
            new_name = '.'.join(new_parts[:4]) + '.' + '.'.join(pre_parts[2:])
            new_name = new_name.replace('intra_inter', 'sub_intra_inter')
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

        # intra & inter decoder
        elif 'intra' in new_parts[3] and 'sub_intra_inter' not in new_module_name and pre_parts[1] == 'decoder' and int(pre_parts[2].split('_')[-1]) % 2 == 1 and 'layer' not in pre_parts[2].split('_')[-2]:
            new_model_dict = new_model.state_dict()

            # intra
            new_name = new_parts[0] + '.' + 'paired_decoder' + '.' + '.'.join(new_parts[1:4]).replace('paired', 'paired_de') + '.' + '.'.join(pre_parts[3:])
            new_name = new_name.replace('_' + new_name.split('.')[2].split('_')[-1], '_' + pre_parts[2].split('_')[-1])
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

            # sub_intra
            new_name = new_parts[0] + '.' + 'paired_decoder' + '.' + '.'.join(new_parts[1:4]).replace('paired', 'paired_de') + '.' + '.'.join(pre_parts[3:])
            new_name = new_name.replace('_' + new_name.split('.')[2].split('_')[-1], '_' + pre_parts[2].split('_')[-1])
            new_name = new_name.replace('intra_inter', 'sub_intra_inter')
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

        elif 'inter' in new_parts[3] and 'sub_intra_inter' not in new_module_name and pre_parts[1] == 'decoder' and int(pre_parts[2].split('_')[-1]) % 2 == 0 and 'layer' not in pre_parts[2].split('_')[-2]:
            new_model_dict = new_model.state_dict()

            # inter
            new_name = new_parts[0] + '.' + 'paired_decoder' + '.' + '.'.join(new_parts[1:4]).replace('paired', 'paired_de') + '.' + '.'.join(pre_parts[3:])
            new_name = new_name.replace('_' + new_name.split('.')[2].split('_')[-1], '_' + pre_parts[2].split('_')[-1])
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

            # sub_inter
            new_name = new_parts[0] + '.' + 'paired_decoder' + '.' + '.'.join(new_parts[1:4]).replace('paired', 'paired_de') + '.' + '.'.join(pre_parts[3:])
            new_name = new_name.replace('_' + new_name.split('.')[2].split('_')[-1], '_' + pre_parts[2].split('_')[-1])
            new_name = new_name.replace('intra_inter', 'sub_intra_inter')
            new_model_dict[new_name] = pre_trained_param
            new_model.load_state_dict(new_model_dict)

        elif pre_parts[1] == 'decoder' and 'de_layer' in new_module_name:
            new_model_dict = new_model.state_dict()
            new_model.load_state_dict(new_model_dict)

    return new_model

