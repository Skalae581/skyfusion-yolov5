# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:47:38 2025

@author: TAKO
"""

# train_yolov5_skyfusion.py
# Autor: TAKO
# Trainiert YOLOv5 auf SkyFusion-Datensatz (Windows-kompatibel mit Multiprocessing)

import os
import sys
from pathlib import Path

def main():
    # ============ [1] YOLOv5 Pfad vorbereiten ============
    yolov5_path = Path("C:/Users/kosch/PycharmProjects/Deep Learning/Projekt/yolov5")
    if not yolov5_path.exists():
        raise FileNotFoundError(f"❌ YOLOv5-Ordner nicht gefunden: {yolov5_path}")
    
    sys.path.append(str(yolov5_path))  # train.py importierbar machen

    from train import run  # ← train.py in YOLOv5 muss vorhanden sein

    # ============ [2] Datenpfade ============
    project_root = Path("C:/Users/kosch/PycharmProjects/Deep Learning/Projekt/YOLO_FORMAT")
    yaml_path = project_root / "skyfusion.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"❌ data.yaml nicht gefunden: {yaml_path}")

    # ============ [3] Training starten ============
    print("🚀 YOLOv5 Training wird gestartet...\n")
    run(
        imgsz=640,                     # Bildgröße
        batch=16,                      # Batch-Größe
        epochs=50,                     # Anzahl Epochen
        data=str(yaml_path),           # Pfad zur YAML
        weights="yolov5n.pt",          # Pretrained Weights (nano-Modell)
        name="skyfusion_yolov5",       # Name des Laufs
        project="runs/train",          # Basisordner
        exist_ok=True                  # nicht abbrechen wenn Ordner existiert
    )

    print("\n✅ Training abgeschlossen. Modell unter 'runs/train/skyfusion_yolov5/' verfügbar.")

# ============ [4] Windows-Schutz für multiprocessing ============
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # nur auf Windows nötig
    main()
