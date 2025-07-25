Hier ist ein vollständiger Vorschlag für eine `README.md`-Datei deines Projekts **YOLOv5 Objekterkennung mit dem SkyFusion-Datensatz**, basierend auf den Informationen aus deiner Präsentation und deinen Python-Skripten:

---

```markdown
# 🛰️ YOLOv5 Objekterkennung auf dem SkyFusion-Datensatz

**Autorin**: Tanja Koschevnikov  
**Projektziel**: Tiny Object Detection auf Satellitenbildern mit YOLOv5

---

## 📁 Projektübersicht

Dieses Projekt verwendet den **SkyFusion-Datensatz** (ein Subset von AI-TOD v2 und Airbus Aircraft Detection), um ein YOLOv5-Modell für die Erkennung von Objekten wie Flugzeuge, Schiffe und Fahrzeuge zu trainieren.  
Die Annotationen lagen ursprünglich im **COCO-Format** vor und wurden mit **pylabel** in das **YOLO-Format** konvertiert.

---

## 🛠️ Verwendete Tools & Umgebung

- **YOLOv5** (Ultralytics, PyTorch-basiert)
- **Kaggle Notebook** mit 2x NVIDIA T4 GPUs
- **Spyder IDE** für lokale Datenkonvertierung
- **Python-Bibliotheken**: `pandas`, `pylabel`, `matplotlib`, `json`, `shutil`

---

## 🗂️ Projektstruktur

```

skyfusion\_yolo/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
├── images/
└── labels/

````

---

## 📦 Datenvorverarbeitung

Die COCO-Annotationen werden mit dem Skript [`V2_yolo.py`](./V2_yolo.py) in das YOLO-Format überführt. Dies umfasst:

- **Einlesen** der COCO-JSON-Dateien
- **Umbenennen** der Spalten
- **Normalisierung** der Bounding Boxes
- **Export** der YOLO-Annotationen und Bilder in die Verzeichnisstruktur

```python
# Beispiel: YOLO-Annotation
class_id x_center y_center width height
````

---

## 🧠 Modelltraining

Das Training erfolgt auf **Kaggle**, basierend auf YOLOv5s mit:

* 📸 Image size: `640x640`
* 🔁 Batch size: `32`
* 🧮 Epochen: `20`, `50`, `80` (beste Ergebnisse bei 80)
* 📂 YAML-Konfiguration mit 4 Klassen: `['unknown', 'aircraft', 'ship', 'vehicle']`

---

## 📊 Auswertung & Visualisierung

Verwendetes Auswertungsskript: [`YOLO Plott.py`](./YOLO%20Plott.py)
Enthält:

* 📉 **Trainings-/Validierungs-Loss** (box, objectness, class)
* 📈 **Metriken**: Precision, Recall, mAP\@0.5, mAP\@0.5:0.95
* Auswahl von **Epoche 122–201** (entspricht den letzten 80 Epochen)

```python
plt.plot(df['metrics/mAP_0.5'], label='mAP@0.5')
plt.plot(df['metrics/recall'], label='Recall')
```

---

## 🧪 Ergebnisse

| Klasse   | F1-Score (Peak) | Besonderheit                   |
| -------- | --------------- | ------------------------------ |
| aircraft | \~0.93          | Sehr hohe Genauigkeit          |
| ship     | \~0.35          | Mittelmäßig, stabil            |
| vehicle  | \~0.2           | Schwächen bei kleinen Objekten |

* Gesamter F1-Peak bei \~**0.49**, erreicht bei einer Konfidenz von **0.234**

---

## ✅ Fazit

* ✔ Erfolgreiche Konvertierung und Training
* ✔ Sehr gute Erkennung bei „aircraft“
* ⚠ Schwächen bei „vehicle“ (klein, wenig annotiert)
* ✔ Stabiles Training dank Kaggle-T4-Umgebung

---

## 📎 Weitere Dateien

| Datei                | Funktion                                 |
| -------------------- | ---------------------------------------- |
| `DL Sky Fusion.pptx` | Präsentation mit Überblick & Ergebnissen |
| `V2_yolo.py`         | Datensatz-Konvertierung COCO → YOLO      |
| `YOLO Plott.py`      | Visualisierung der Trainingsmetriken     |

---

## 📌 To Do (optional)

* Hyperparameter-Tuning für schwierige Klassen
* Verwendung größerer YOLOv5-Modelle (YOLOv5m, YOLOv5l)
* Datenerweiterung & Label-Validierung

---

**Vielen Dank fürs Lesen!**

```

---

Möchtest du eine `.md`-Datei exportiert bekommen? Ich kann sie dir direkt als Datei erstellen.
```
