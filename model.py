import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.nn.functional.relu6(self.bn1(self.depthwise(x)))
        x = torch.nn.functional.relu6(self.bn2(self.pointwise(x)))
        return x
class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # input size: 32x32x3, output size: 16x16x32
            DepthwiseSeparableConv(32, 64, 1),  # output size: 16x16x64
            DepthwiseSeparableConv(64, 128, 2), # output size: 8x8x128
            DepthwiseSeparableConv(128, 128, 1),# output size: 8x8x128
            DepthwiseSeparableConv(128, 256, 2),# output size: 4x4x256
            DepthwiseSeparableConv(256, 256, 1),# output size: 4x4x256
            DepthwiseSeparableConv(256, 512, 2),# output size: 2x2x512
            DepthwiseSeparableConv(512, 512, 1),# 5x repeated, output size: 2x2x512
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),# output size: 1x1x1024
            DepthwiseSeparableConv(1024, 1024, 1),# output size: 1x1x1024
            nn.AdaptiveAvgPool2d(1), # output size: 1x1x1024
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# Khởi tạo mô hình
model = MobileNet(num_classes=2)  # Chuyển mô hình sang GPU nếu có

class CroatianFishClassifier(pl.LightningModule):
    def __init__(self, num_classes = 2):
        super(CroatianFishClassifier, self).__init__()
        self.model = model
        self.train_accuracies = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        out = correct / total
        return out

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}

# Kiểm tra checkpoint tốt nhất
best_checkpoint_path = "/Users/phamminhtuan/Desktop/AIChallenge/lightning_logs/version_8/checkpoints/best-checkpoint.ckpt"
if best_checkpoint_path:
    print(f"Loading best checkpoint from: {best_checkpoint_path}")
    
    # Load mô hình từ checkpoint (gọi trực tiếp trên class)
    model = CroatianFishClassifier.load_from_checkpoint(best_checkpoint_path)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    model.freeze()
else:
    print("No best checkpoint found!")
def Model(model = model):
    return model