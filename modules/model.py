import torch
from torchvision import models as models
from torch import nn
class DigitResNet18(nn.Module):
    # 数字识别模型，自己训练的，可能PaddleOCR 更好用
    def __init__(self, num_classes=9):
        super(DigitResNet18, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(weights, map_location='cpu')
    detect_model = DigitResNet18()
    detect_model.load_state_dict(state_dict)
    detect_model.to(device)
    detect_model.eval()  # 设置为评估模式
    return detect_model
