# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:34:33 2025

@author: TAKO
"""

# train_yolov5_direct.py

import os
from pathlib import Path
import sys

# =====================================
# 1. Pfade definieren
# =====================================
project_root = Path("C:/Users/kosch/PycharmProjects/Deep Learning/Projekt/YOLO_FORMAT")
train_dir = project_root / "train" / "images"
val_dir = project_root / "valid" / "images"
yaml_path = project_root / "skyfusion.yaml"

# =====================================
# 2. YAML-Datei erzeugen
# =====================================
yaml_content = f"""
train: {train_dir.as_posix()}
val: {val_dir.as_posix()}

nc: 4
names: ['vehicle', 'ship', 'aircraft','unknown']
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content.strip())

print(f"✅ YAML gespeichert unter: {yaml_path}")

# =====================================
# 3. YOLOv5 direkt importieren & starten
# =====================================
# Sicherstellen, dass yolov5 im Pfad ist
yolov5_path = Path("C:/Users/kosch/PycharmProjects/Deep Learning/Projekt/yolov5")
if not yolov5_path.exists():
    raise FileNotFoundError(f"❌ YOLOv5-Ordner nicht gefunden: {yolov5_path}")

sys.path.append(str(yolov5_path))

from train import run  # Direkt das Trainingsmodul aus YOLOv5 importieren

from pathlib import Path
import sys

# Optional: Logging, Path-Setup etc.

def main():
    from train import run

    project_root = Path("C:/Users/kosch/PycharmProjects/Deep Learning/Projekt/YOLO_FORMAT")
    yaml_path = project_root / "skyfusion.yaml"

    run(
        imgsz=640,
        batch=16,
        epochs=50,
        data=str(yaml_path),
        weights="yolov5n.pt",
        name="skyfusion_yolov5",
        project="runs/train",
        exist_ok=True
    )

# =======================================
# Wichtig für Windows:
# Multiprocessing benötigt dies!
# =======================================
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # optional, aber sicher
    main()
