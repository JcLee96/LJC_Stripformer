import os
import cv2
import numpy as np
from pathlib import Path

def save_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

root = Path("./dummy")

# 폴더 구조 (train/test만 있어도 됨)
blur_train_dir  = root / "frames_avg/100k/train"
sharp_train_dir = root / "frames/100k/train"
blur_test_dir   = root / "frames_avg/100k/test"
sharp_test_dir  = root / "frames/100k/test"

H, W = 720, 1280

# base sharp
sharp = np.zeros((H, W, 3), dtype=np.uint8)
cv2.putText(sharp, "SHARP_000000", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5, cv2.LINE_AA)
cv2.rectangle(sharp, (200, 300), (900, 650), (0, 255, 0), -1)

# blur: 가우시안 블러로 생성
blur = cv2.GaussianBlur(sharp, (21, 21), 0)

# sub(+2): 일단 sharp 복사해서 텍스트만 바꿔도 OK
sub = sharp.copy()
cv2.putText(sub, "SUB_000002", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5, cv2.LINE_AA)

# train
save_png(blur_train_dir / "000000.png", blur)
save_png(sharp_train_dir / "000000.png", sharp)
save_png(sharp_train_dir / "000002.png", sub)   # +2

# test
save_png(blur_test_dir / "000000.png", blur)
save_png(sharp_test_dir / "000000.png", sharp)
save_png(sharp_test_dir / "000002.png", sub)    # +2

print("DONE: dummy data created at ./dummy")
