import torch.nn as nn
from models.Stripformer import Stripformer
from models.Stripformer_gru_t import StripformerMultiInputV2 as GRU_model
from models.Stripformer_cross_att import StripformerMultiInputV2 as Cross_att_model


def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer()
    elif generator_name == 'Stripformer_GRU_t':
        model_g = GRU_model(mode=model_config['mode'])
    elif generator_name == 'Stripformer_cross_att':
        model_g = Cross_att_model(mode=model_config['mode'])

    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)


def get_nets(model_config):
    return get_generator(model_config)
