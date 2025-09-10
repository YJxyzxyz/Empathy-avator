# models/fer/sanity_infer.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from models.fer.infer import FER, DEFAULT_CLASSES  # 复用同一套预处理/推理

FER_ONNX = "models/fer/weights/fer_mbv3.onnx"  # 按你的实际路径
IMG_SIZE = (224, 224)
CLASSES = DEFAULT_CLASSES  # ('angry','disgust','fear','happy','sad','surprise','neutral')

def main():
    fer = FER(FER_ONNX, input_size=IMG_SIZE, classes=CLASSES)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头"); return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = fer.infer(frame)
        probs = out["probs"]
        label = CLASSES[int(np.argmax(probs))]
        conf  = float(np.max(probs))
        bbox  = out.get("bbox")

        # 画框与文字
        vis = frame.copy()
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("FER Sanity (via FER class)", vis)
        if cv2.waitKey(1) == 27:  # Esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
