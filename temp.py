import torch

layer = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2)

dummy_input = torch.FloatTensor(torch.randn((1, 3, 64, 64)))

print(layer(dummy_input).shape)