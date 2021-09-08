from torch.nn.init import init
from torch.nn import nn

def initNetParams(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=1e-2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)