# Fruit Vision

## Voraussetzungen
Eine vom Geräte-Manager gefundene Kamera ist notwendig für die Nutzung des Skripts.

Installiere [Python 3](https://www.python.org/downloads/) und die benötigten Pakete:

```bash
python -m pip install PyQt6 opencv-python ultralytics
```

## Anwendung starten

Führe den Befehl im Projektordner aus:

```bash
python main.py
```

Die Anwendung benötigt eine angeschlossene Kamera.

## Modell-Dateien

Die drei trainierten Modelle müssen im Projekt unter den folgenden Pfaden liegen:

- `models/trained/Object_detection/weights/trained_object_detection_1.pt`
- `models/trained/Classification_1_own_data/weights/best.pt`
- `models/trained/Segmentation meine Daten/weights/best.pt`

Die Pfade werden relativ zum Projektverzeichnis aufgelöst. Dadurch funktionieren sie unabhängig davon, von welchem Ordner aus du `main.py` startest. Wenn die Gewichte beim Hochladen nicht mit übertragen werden, lade die Dateien herunter oder lege sie an den oben genannten Speicherorten ab.
