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


## Genutzte Datensätze

Fruit Detection Dataset von Lakshay Tyagi · Nishita Gunjal
```bash
https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/data
```

Fruit-360 von Mihai Oltean 
```bash
https://www.kaggle.com/datasets/moltean/fruits
```


## Decleration of AI-Usage
In dem gesamten Projekt wurde für die Aufgaben, das erstellen des GUIs, trainieren der Modelle, sowie Einbindung in die Kameraanwendung, die KI ausschließlich als "Tutor" und selten zum debuggen genutzt. Die einzigen von KI generierten Skripts, sind die Vergleichsauswertungen der jeweiligen Modelle und das Skript, um Fotos im gleichen Format auf beiden meiner Geräte zu machen.

Prompts sahen nach der Anweisung keinen Code-Vorschlag zu bekommen oder direkte Änderungen vorgenommen werden strikt so aus:

```bash
Ich arbeite gerade an meinem Projekt über ... gebe mir eine grobe vorgehens Struktur.
```

```bash
Welche Pakete/Frameworks sollte ich zu Realisierung meiner Ideen nutzen?
```

```bash
Was sollten die nächsten Schritte in meiner Projektarbeit sein?
```

```bash
Was bedeuten die Parameter in der .train(...) Funktion? 
```

```bash
Ich möchte für mein GUI eine transparente Ebene haben, was für ein Layout brauche ich dafür?
```

```bash
Auf was sollte ich beim aufnehmen der Fotos für meinen Datensatz achten, damit das Ergebnis möglichst vielversprechend wird
```
