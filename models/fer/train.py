import os, sys, time, math, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
from dataset_fer2013 import build_datasets

FER7 = ["angry","disgust","fear","happy","sad","surprise","neutral"]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # (N,)
        pt = torch.exp(-ce)
        loss = (1-pt)**self.gamma * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def build_model(name="mobilenetv3_large", num_classes=7, pretrained=True):
    if name == "mobilenetv3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("unknown model")
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1,total)

def train(
    data_root="data/processed/FER2013_Images",
    out_ckpt="models/fer/weights/best.ckpt",
    model_name="mobilenetv3_large",
    img_size=224,
    epochs=40,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    loss_type="focal",   # "ce" or "focal"
    patience=6,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, test_ds = build_datasets(data_root, img_size)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(model_name, num_classes=len(FER7), pretrained=True).to(device)

    # 类别不均衡时可设置alpha，例如根据train集类别频次的反比
    criterion = nn.CrossEntropyLoss() if loss_type=="ce" else FocalLoss(gamma=2.0, alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best_acc, best_state = 0.0, None
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()

        # 验证
        val_acc = evaluate(model, val_ld, device)
        train_loss = total_loss / len(train_ds)
        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] loss={train_loss:.4f} val_acc={val_acc:.4f} time={dt:.1f}s")

        # 早停/保存
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    # 保存最佳
    Path(os.path.dirname(out_ckpt)).mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "meta":{
        "classes": FER7, "model_name": model_name, "img_size": img_size
    }}, out_ckpt)
    print("Best val_acc:", best_acc, "saved to", out_ckpt)

    # 最终在test集上评估
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_ld, device)
    print("Test acc:", test_acc)
    return out_ckpt

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/processed/FER2013_Images")
    ap.add_argument("--out_ckpt",  default="models/fer/weights/best.ckpt")
    ap.add_argument("--model_name", default="mobilenetv3_large", choices=["mobilenetv3_large","mobilenetv3_small","resnet18"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss_type", default="focal", choices=["ce","focal"])
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    train(**vars(args))
