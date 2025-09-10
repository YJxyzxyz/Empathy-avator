import os, csv, cv2, numpy as np
from pathlib import Path

# FER2013官方7类：0=angry,1=disgust,2=fear,3=happy,4=sad,5=surprise,6=neutral
FER7 = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def write_img(root, split, label, idx, pixels, size=48):
    arr = np.asarray([int(p) for p in pixels.split()], dtype=np.uint8).reshape(size, size)
    # 转成3通道，后续训练统一走RGB
    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    out_dir = Path(root)/split/FER7[int(label)]
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir/f"{idx:07d}.jpg"), img)

def main():
    csv_path = Path("data/raw/fer2013.csv")
    out_root = Path("data/processed/FER2013_Images")
    out_root.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        idx_train = idx_val = idx_test = 0
        for row in reader:
            label = int(row["emotion"])
            pixels = row["pixels"]
            usage = row["Usage"].lower()
            if "training" in usage:
                write_img(out_root, "train", label, idx_train, pixels)
                idx_train += 1
            elif "publictest" in usage:
                write_img(out_root, "val", label, idx_val, pixels)
                idx_val += 1
            else:  # "PrivateTest"
                write_img(out_root, "test", label, idx_test, pixels)
                idx_test += 1
    print("Done. Output at:", out_root)

if __name__ == "__main__":
    main()
