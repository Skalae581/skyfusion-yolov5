import numpy as np
import pandas as pd
import cv2
from PIL import Image
import json
import sys
import os
import shutil
import os
import json
import imagesize
from pylabel import importer
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
import re
import yaml
from shutil import copyfile
import pylabel
import importlib.metadata

from pylabel import importer



# ==============================
# Pfade definieren
# ==============================
train_path = r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\train'
valid_path = r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\valid'
test_path  = r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\SkyFusion\test'

train_annots = f'{train_path}\\_annotations.coco.json'
valid_annots = f'{valid_path}\\_annotations.coco.json'
test_annots  = f'{test_path}\\_annotations.coco.json'

destination_path = r'C:\Users\kosch\PycharmProjects\Deep Learning\Projekt\YOLO_FORMAT'
os.makedirs(destination_path, exist_ok=True)

# ==============================
# Dienstklasse
# ==============================
class ScriptUtilities:
    @staticmethod
    def json_viewer(annots_json):
        with open(annots_json, 'r') as file:
            annotations_data = json.load(file)
        
        for image_info in annotations_data['images'][:5]:
            print(f"Image ID: {image_info['id']}")
            print(f"File Name: {image_info['file_name']}")
            print(f"Height: {image_info['height']}, Width: {image_info['width']}")
            print(f"Date Captured: {image_info.get('date_captured', 'N/A')}")
            print("----------------------")

        if 'annotations' in annotations_data:
            print("Annotations:")
            for annotation in annotations_data['annotations'][:5]:
                print(annotation)
                print("----------------------")
        else:
            print("No 'annotations' key found in the JSON data.")
    
    @staticmethod
    def json_tree(annots_json):
        with open(annots_json, 'r') as file:
            annotations_data = json.load(file)
        print(f"Top-level keys in JSON: {list(annotations_data.keys())}")

    @staticmethod
    def coco_to_yolo_converter(annots_json, path_to_images, name):
        dataset = importer.ImportCoco(annots_json, path_to_images=path_to_images, name=name)
        return dataset

# ==============================
# JSON-Dateien inspizieren
# ==============================
json_files = {'train': train_annots, 'valid': valid_annots, 'test': test_annots}
utilities = ScriptUtilities()

for title, file_path in json_files.items():
    print(f"\n--- {title.upper()} JSON TREE ---")
    utilities.json_tree(file_path)
    print(f"\n--- {title.upper()} JSON VIEWER ---")
    utilities.json_viewer(file_path)

# ==============================
# COCO zu YOLO konvertieren
# ==============================
train_dataset = ScriptUtilities.coco_to_yolo_converter(train_annots, train_path, "train_set")
valid_dataset = ScriptUtilities.coco_to_yolo_converter(valid_annots, valid_path, "valid_set")
test_dataset  = ScriptUtilities.coco_to_yolo_converter(test_annots, test_path, "test_set")

# Beispiel: Zeige zufällige 5 Annotationen aus dem Trainingsset
print(train_dataset.df.sample(5))
print("\n📋 Spaltenübersicht im train_dataset:")
print(train_dataset.df.columns.tolist())
# Spalten umbenennen für YOLO-kompatiblen Export
train_dataset.df = train_dataset.df.rename(columns={
    'bbox_x': 'x',
    'bbox_y': 'y',
    'bbox_width': 'width',
    'bbox_height': 'height',
    'category_id': 'class_id',
    'image_path': 'img_path',         # nur falls vorhanden
    'image_filename': 'img_filename'  # nur falls vorhanden
})
valid_dataset.df = valid_dataset.df.rename(columns=train_dataset.df.columns.to_series().to_dict())
test_dataset.df = test_dataset.df.rename(columns=train_dataset.df.columns.to_series().to_dict())

# ==============================
# Export in YOLOv5-kompatibles Format
# ==============================
#train_dataset.export.ExportToYolo(output_path=os.path.join(destination_path, 'train'), cat_id_index=0)
#valid_dataset.export.ExportToYolo(output_path=os.path.join(destination_path, 'valid'), cat_id_index=0)
#test_dataset.export.ExportToYolo(output_path=os.path.join(destination_path, 'test'), cat_id_index=0)
# Exportiere Trainingsdaten ins YOLOv5-Format
import os
import shutil

def export_dataset_to_yolo(dataset, output_path, image_folder_name="images", label_folder_name="labels"):
    images_path = os.path.join(output_path, image_folder_name)
    labels_path = os.path.join(output_path, label_folder_name)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Gruppiere nach Bild-ID, damit alle Objekte gemeinsam geschrieben werden
    grouped = dataset.df.groupby('img_filename')

    for filename, group in grouped:
        # Zielpfade
        image_path = group.iloc[0]['img_path']
        label_path = os.path.join(labels_path, os.path.splitext(filename)[0] + '.txt')
        image_target_path = os.path.join(images_path, os.path.basename(image_path))

        # Bild kopieren (nur einmal)
        try:
            if not os.path.exists(image_target_path):
                shutil.copy(image_path, image_target_path)
        except Exception as e:
            print(f"⚠️ Fehler beim Kopieren von {image_path}: {e}")
            continue

        # Schreibe alle Objekte für dieses Bild in eine .txt-Datei
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                x, y, w, h = row['x'], row['y'], row['width'], row['height']
                img_w, img_h = row['img_width'], row['img_height']
                class_id = row['class_id']

                # Normalisierung
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # YOLO-Zeile
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                f.write(yolo_line + '\n')

    print(f"✅ Export abgeschlossen: {len(grouped)} Bilder verarbeitet → {output_path}")

export_dataset_to_yolo(train_dataset, os.path.join(destination_path, 'train'))
export_dataset_to_yolo(valid_dataset, os.path.join(destination_path, 'valid'))
export_dataset_to_yolo(test_dataset, os.path.join(destination_path, 'test'))

print("✅ Export ins YOLOv5-Format erfolgreich abgeschlossen.")
