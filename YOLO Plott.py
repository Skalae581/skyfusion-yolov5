# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:30:22 2025

@author: TAKO
"""

import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei
csv_path = r"C:\Users\kosch\Downloads\results (2).csv"

# CSV laden
df = pd.read_csv(csv_path)

# Neue Spalte 'id_epoche' hinzufügen (beginnend bei 1)
df['id_epoche'] = range(1, len(df) + 1)

# Neue CSV-Datei speichern (optional: überschreiben)
df.to_csv(csv_path, index=False)  # oder 'results_with_id.csv' für neue Datei

print(df.head())  # Zum Prüfen
# Spaltennamen bereinigen
df.columns = df.columns.str.strip()

# Kontrollausgabe
print(df.columns.tolist())

# Nur die Epochen 0–79 auswählen
df_80 = df.iloc[122:201]

# 📉 Plot: Trainings- und Validierungsverluste
plt.figure(figsize=(16, 4))
plt.plot(df_80['train/box_loss'], label='Box Loss')
plt.plot(df_80['train/obj_loss'], label='Objectness Loss')
plt.plot(df_80['train/cls_loss'], label='Class Loss')
plt.plot(df_80['val/box_loss'], label='Val Box Loss', linestyle='--')
plt.plot(df_80['val/obj_loss'], label='Val Objectness Loss', linestyle='--')
plt.plot(df_80['val/cls_loss'], label='Val Class Loss', linestyle='--')
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Trainings- und Validierungsverluste (122-201)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📈 Plot: Precision, Recall, mAP
plt.figure(figsize=(12, 6))
plt.plot(df_80['metrics/precision'], label='Precision', color='blue')
plt.plot(df_80['metrics/recall'], label='Recall', color='green')
plt.plot(df_80['metrics/mAP_0.5'], label='mAP@0.5', color='orange')

# Optional: mAP@0.5:0.95, falls vorhanden
if 'metrics/mAP_0.5:0.95' in df_80.columns:
    plt.plot(df_80['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='red', linestyle='--')

plt.title("Lernkurven: Precision, Recall, mAP (122-201)")
plt.xlabel("Epoche")
plt.ylabel("Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
