# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:37:38 2025

@author: TAKO
"""

import os
import json
import shutil
import pandas as pd
from pylabel import importer

# ==============================
# Konfigurierbare Pfade
# ==============================
data_paths = {
    'train': r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\train',
    'valid': r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\valid',
    'test':  r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\test'
}

annots_files = {k: os.path.join(v, '_annotations.coco.json') for k, v in data_paths.items()}
destination_path = r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\YOLO_FORMAT'
os.makedirs(destination_path, exist_ok=True)

# ==============================
# Dienstklasse
# ==============================
class ScriptUtilities:
    @staticmethod
    def json_viewer(annots_json):
        with open(annots_json, 'r') as file:
            data = json.load(file)
        for img in data['images'][:5]:
            print(f"Image ID: {img['id']}\nFile Name: {img['file_name']}\nHeight: {img['height']}, Width: {img['width']}\nDate Captured: {img.get('date_captured', 'N/A')}\n{'-'*22}")
        if 'annotations' in data:
            print("Annotations:")
            for ann in data['annotations'][:5]:
                print(ann)
                print("-"*22)

    @staticmethod
    def json_tree(annots_json):
        with open(annots_json, 'r') as file:
            data = json.load(file)
        print(f"Top-level keys: {list(data.keys())}")

    @staticmethod
    def coco_to_yolo_converter(annots_json, path_to_images, name):
        return importer.ImportCoco(annots_json, path_to_images=path_to_images, name=name)

# ==============================
# JSON inspizieren
# ==============================
utilities = ScriptUtilities()
for name, path in annots_files.items():
    print(f"\n--- {name.upper()} JSON TREE ---")
    utilities.json_tree(path)
    print(f"\n--- {name.upper()} JSON VIEWER ---")
    utilities.json_viewer(path)

# ==============================
# Konvertierung COCO → YOLO mit pylabel
# ==============================
datasets = {
    name: ScriptUtilities.coco_to_yolo_converter(annots_files[name], data_paths[name], f"{name}_set")
    for name in ['train', 'valid', 'test']
}

# Spalten umbenennen
for ds in datasets.values():
    ds.df = ds.df.rename(columns={
        'bbox_x': 'x', 'bbox_y': 'y', 'bbox_width': 'width', 'bbox_height': 'height',
        'category_id': 'class_id', 'image_path': 'img_path', 'image_filename': 'img_filename'
    })
print(ds.df.columns)

# ==============================
# Exportfunktion
# ==============================
def export_dataset_to_yolo(dataset, output_path, image_folder="images", label_folder="labels"):
    import os
    import shutil

    df = dataset.df.copy()

    # 🛠 Spalten vorbereiten, falls noch nicht vorhanden
    if 'x' not in df.columns or 'y' not in df.columns:
        df['x'] = df['ann_bbox_xmin']
        df['y'] = df['ann_bbox_ymin']
        df['width'] = df['ann_bbox_width']
        df['height'] = df['ann_bbox_height']
        df['class_id'] = df['cat_id']

    img_path = os.path.join(output_path, image_folder)
    lbl_path = os.path.join(output_path, label_folder)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(lbl_path, exist_ok=True)

    grouped = df.groupby('img_filename')

    for filename, group in grouped:
        image_src = group.iloc[0]['img_path']
        image_dst = os.path.join(img_path, os.path.basename(image_src))
        label_file = os.path.join(lbl_path, os.path.splitext(filename)[0] + '.txt')

        try:
            if not os.path.exists(image_dst):
                shutil.copy(image_src, image_dst)
        except Exception as e:
            print(f"⚠️ Fehler beim Kopieren von {image_src}: {e}")
            continue

        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                iw, ih = row['img_width'], row['img_height']
                class_id = int(row['class_id'])  # Sicherheitshalber in int

                # Normalisierung (YOLO-Format)
                xc = (x + w / 2) / iw
                yc = (y + h / 2) / ih
                wn = w / iw
                hn = h / ih

                f.write(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    print(f"✅ {len(grouped)} Dateien nach {output_path} exportiert")


# ==============================
# Export starten
# ==============================
for name, ds in datasets.items():
    export_dataset_to_yolo(ds, os.path.join(destination_path, name))

print("\n✅ YOLO-Export abgeschlossen.")