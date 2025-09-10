import os, torch, torch.nn as nn
from torchvision import models
from pathlib import Path
from train import build_model, FER7

def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    return ckpt.get("meta", {})

def export_onnx(ckpt="models/fer/weights/best.ckpt",
                out_path="models/fer/weights/fer_mbv3.onnx",
                model_name="mobilenetv3_large",
                img_size=224):
    device = torch.device("cpu")
    model = build_model(model_name, num_classes=len(FER7), pretrained=False).to(device)
    meta = load_ckpt(model, ckpt)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0:"N"}, "logits": {0:"N"}}
    )
    print("Exported to", out_path)

if __name__ == "__main__":
    export_onnx()
