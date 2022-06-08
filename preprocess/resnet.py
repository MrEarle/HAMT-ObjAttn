import torch
from torch import Tensor


def _forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


def _forward_impl_2(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def get_model():
    # conv = None

    # def get_conv():
    #     nonlocal conv
    #     return conv

    # def hook(module, ins, outs: torch.Tensor):
    #     nonlocal conv
    #     conv = outs.cpu().detach().numpy()

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)
    model._forward_impl = lambda x: _forward_impl(model, x)
    # model.layer4.register_forward_hook(hook)
    model.eval()
    return model  # , get_conv


def get_model_pooled():
    # conv = None

    # def get_conv():
    #     nonlocal conv
    #     return conv

    # def hook(module, ins, outs: torch.Tensor):
    #     nonlocal conv
    #     conv = outs.cpu().detach().numpy()

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)
    model._forward_impl = lambda x: _forward_impl_2(model, x)
    # model.layer4.register_forward_hook(hook)
    model.eval()
    return model  # , get_conv
