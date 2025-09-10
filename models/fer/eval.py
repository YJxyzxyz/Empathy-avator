import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ---------- 构建模型骨架，与 train.py 对齐 ----------
def build_model(model_name="mobilenetv3_large", num_classes=7, pretrained=False):
    if model_name == "mobilenetv3_large":
        m = models.mobilenet_v3_large(weights=None if not pretrained else models.MobileNet_V3_Large_Weights.DEFAULT)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif model_name == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=None if not pretrained else models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif model_name == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return m


# ---------- 解析 ckpt，拿到真正的 state_dict ----------
def load_model(ckpt_path, device="cuda", fallback_model_name="mobilenetv3_large", fallback_num_classes=7):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    meta = {}
    state_dict = None

    if isinstance(ckpt, dict):
        # 我们在 train.py 是保存 {"model": best_state, "meta": {...}}
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
            meta = ckpt.get("meta", {})
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
            meta = ckpt.get("meta", {})
        else:
            # 可能直接是纯 state_dict
            # 经验判断：包含若干卷积/BN键名即认为是权重
            some_keys = list(ckpt.keys())[:5]
            if all(isinstance(k, str) for k in ckpt.keys()):
                state_dict = ckpt
            else:
                raise RuntimeError("ckpt 格式不认识：既不是{'model': state_dict}也不是{'state_dict': ...}，也不是纯 state_dict")
    else:
        # 直接保存了整个模型对象（不推荐）
        model = ckpt.to(device)
        model.eval()
        return model, meta

    # 从 meta 里还原模型信息
    model_name = meta.get("model_name", fallback_model_name)
    classes = meta.get("classes", None)
    num_classes = len(classes) if isinstance(classes, (list, tuple)) else fallback_num_classes

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("[警告] 有缺失权重未加载（通常是分类层不匹配时出现）:")
        print("  missing:", missing[:10], ("...(+%d)" % (len(missing)-10) if len(missing) > 10 else ""))
    if unexpected:
        print("[警告] 有多余键在 ckpt 中未被模型使用（可能是保存了多余内容）:")
        print("  unexpected:", unexpected[:10], ("...(+%d)" % (len(unexpected)-10) if len(unexpected) > 10 else ""))

    model.to(device).eval()
    return model, meta


def build_eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def run_eval(model, dataloader, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    all_preds, all_labels = [], []

    for imgs, labels in tqdm(dataloader, desc="Evaluating"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, total_samples)
    return acc, avg_loss, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(cm, class_names, out_png="fer_confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[信息] 混淆矩阵图已保存：{out_png}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = build_eval_transform(args.img_size)

    dataset = datasets.ImageFolder(root=args.data_root, transform=tf)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # 加载模型（自动识别我们在 train.py 保存的 {"model": state_dict, "meta": {...}}）
    model, meta = load_model(args.ckpt, device=device,
                             fallback_model_name=args.model_name,
                             fallback_num_classes=args.num_classes)

    # 类别名：优先用 ckpt meta 里的 classes；否则用命令行指定或默认 FER7
    if "classes" in meta and isinstance(meta["classes"], (list, tuple)):
        class_names = list(meta["classes"])
    else:
        # 默认 FER7
        class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    acc, avg_loss, preds, labels = run_eval(model, dataloader, device=device)

    print(f"\n[结果] 集合: {args.data_root}")
    print(f"准确率: {acc*100:.2f}%")
    print(f"平均loss: {avg_loss:.4f}\n")

    print("分类报告:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    print("混淆矩阵:")
    print(cm)

    # 保存混淆矩阵热力图
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = str(out_dir / "fer_confusion_matrix.png")
    save_confusion_matrix(cm, class_names, out_png=out_png, title="FER Confusion Matrix")

    # （可选）保存文本报告
    report_txt = str(out_dir / "eval_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.data_root}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"AvgLoss: {avg_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(labels, preds, target_names=class_names, digits=4))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"[信息] 文本报告已保存：{report_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed/FER2013_Images/test",
                    help="验证/测试集目录（ImageFolder 结构）")
    ap.add_argument("--ckpt", type=str, default="models/fer/weights/best.ckpt",
                    help="训练保存的模型权重路径")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)

    # 作为兜底信息（当 ckpt.meta 里没有记录时使用）
    ap.add_argument("--model_name", type=str, default="mobilenetv3_large",
                    choices=["mobilenetv3_large", "mobilenetv3_small", "resnet18"])
    ap.add_argument("--num_classes", type=int, default=7)

    ap.add_argument("--save_dir", type=str, default="results",
                    help="保存混淆矩阵图和评估文本的目录")
    args = ap.parse_args()
    main(args)
