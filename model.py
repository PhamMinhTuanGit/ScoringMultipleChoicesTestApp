import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import VisionTransformer as ViT

# Khởi tạo mô hình, loss function và optimizer
model = ViT(
            image_size=32,  # Kích thước ảnh đầu vào là 32x32
            patch_size=8,   # Kích thước patch sẽ giảm xuống (kích thước này là tùy chọn)
            num_classes=2,
            hidden_dim=256,        # Kích thước đầu ra của Transformer Encoder (có thể tùy chỉnh)
            num_layers=6,        # Số lớp của Transformer Encoder
            num_heads=8,        # Số attention heads
            mlp_dim=512     # Kích thước của MLP sau attention layers
        )
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
best_checkpoint_path = "/Users/phamminhtuan/Desktop/AIChallenge/lightning_logs/version_15/checkpoints/best-checkpoint.ckpt"
if best_checkpoint_path:
    print(f"Loading best checkpoint from: {best_checkpoint_path}")
    
    # Load mô hình từ checkpoint (gọi trực tiếp trên class)
    model = CroatianFishClassifier.load_from_checkpoint(best_checkpoint_path, weights_only=True)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    model.freeze()
else:
    print("No best checkpoint found!")
def Model(model = model):
    return model