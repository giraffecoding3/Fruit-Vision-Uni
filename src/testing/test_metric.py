# THIS FILE IS WRITTEN BY AI



"""Compare the old and new object-detection models on the held-out test set."""

from __future__ import annotations

import csv
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
import matplotlib
from ultralytics import YOLO

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH_NEW = (
    ROOT
    / "models"
    / "trained"
    / "trained_object_detection"
    / "weights"
    / "trained_object_detection_1.pt"
)
MODEL_PATH_OLD = (
    ROOT
    / "models"
    / "trained"
    / "trained_object_detection"
    / "weights"
    / "vergleich.pt"
)

TEST_DIR = ROOT / "trainingsdata" / "object_detection" / "test"
TRAIN_DIR = ROOT / "trainingsdata" / "object_detection"
CLASS_NAMES = {0: "Apple", 1: "Banana"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 640


def get_test_paths() -> tuple[Path, Path, list[Path]]:
    """Find images and their YOLO labels in either standard or flat layout."""
    images_dir = TEST_DIR / "images" if (TEST_DIR / "images").is_dir() else TEST_DIR
    labels_dir = TEST_DIR / "labels" if images_dir.name == "images" else images_dir

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if images_dir.is_dir() else []

    if not images:
        raise FileNotFoundError(f"Keine Testbilder gefunden in: {images_dir}")

    missing_labels = [image.with_suffix(".txt") for image in images if not (labels_dir / f"{image.stem}.txt").is_file()]
    if missing_labels:
        examples = ", ".join(str(path) for path in missing_labels[:5])
        raise FileNotFoundError(
            f"Für {len(missing_labels)} von {len(images)} Testbildern fehlen YOLO-Labels. "
            "Exportiere die geprüften MakeSense-Labels im YOLO-Format. Lege die .txt-Dateien "
            f"neben die Bilder (flache Ordnerstruktur) oder in {TEST_DIR / 'labels'} "
            f"bei Bildern unter {TEST_DIR / 'images'}. Beispiele: {examples}"
        )

    # Detection labels have: class_id, x_center, y_center, width, height.
    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue  # An empty file is a valid label for an image with no objects.
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: erwartet ein YOLO-Bounding-Box-Label "
                    "mit genau 5 Werten; Polygone/Segmentierungslabels passen nicht."
                )
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"Ungültiges Label in {label_path}:{line_number}") from error
            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unbekannte Klassen-ID {class_id} in {label_path}:{line_number}; "
                    "erwartet werden Apple=0 und Banana=1."
                )
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError(
                    f"Koordinaten außerhalb des normalisierten Bereichs [0, 1] "
                    f"in {label_path}:{line_number}."
                )

    return images_dir, labels_dir, images


def check_model_classes(model: YOLO, weight_path: Path) -> None:
    names = model.names
    ordered_names = [names[index].strip().lower() for index in sorted(names)]
    expected_names = [CLASS_NAMES[index].lower() for index in sorted(CLASS_NAMES)]
    if ordered_names != expected_names:
        raise ValueError(
            f"Klassen in {weight_path.name} sind {ordered_names}; erwartet wird "
            f"{expected_names} in dieser Reihenfolge. Modell und Testlabels müssen "
            "dieselben Klassen-IDs verwenden."
        )


def main() -> None:
    images_dir, _, images = get_test_paths()
    model_paths = {
        "new": MODEL_PATH_NEW,
        "old": MODEL_PATH_OLD,
    }
    for path in model_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Modellgewicht nicht gefunden: {path}")

    # Build a temporary data YAML without changing the project's training YAML.
    data_config = {
        "path": str(ROOT),
        "train": str(TRAIN_DIR / "train" / "images"),
        "val": str(TRAIN_DIR / "valid" / "images"),
        "test": str(images_dir),
        "names": CLASS_NAMES,
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "runs" / "test_metric"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"comparison_{run_id}.csv"
    rows: list[dict[str, object]] = []

    print(f"Testbilder: {len(images)} aus {images_dir}")
    print("Auswertung startet. Beide Modelle erhalten dieselben Bilder und Einstellungen.\n")

    with tempfile.TemporaryDirectory(prefix="test_metric_", dir=ROOT / "src" / "testing") as temp_dir:
        data_yaml = Path(temp_dir) / "test_data.yaml"
        data_yaml.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")

        for label, weight_path in model_paths.items():
            print(f"=== {label.upper()}: {weight_path.name} ===")
            model = YOLO(str(weight_path))
            check_model_classes(model, weight_path)
            metrics = model.val(
                data=str(data_yaml),
                split="test",
                imgsz=IMAGE_SIZE,
                plots=True,
                project=str(results_dir),
                name=f"{run_id}_{label}",
                verbose=True,
            )

            box = metrics.box
            row: dict[str, object] = {
                "model": label,
                "weights": str(weight_path.relative_to(ROOT)),
                "images": len(images),
                "mAP50-95": float(box.map),
                "mAP50": float(box.map50),
                "mAP75": float(box.map75),
                "precision": float(box.mp),
                "recall": float(box.mr),
            }
            per_class_map = box.maps
            for class_id, class_name in CLASS_NAMES.items():
                row[f"mAP50-95 {class_name}"] = float(per_class_map[class_id])
            rows.append(row)

            print(
                f"mAP50-95={row['mAP50-95']:.4f} | mAP50={row['mAP50']:.4f} | "
                f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f}\n"
            )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = results_dir / f"comparison_{run_id}.png"
    save_comparison_plot(rows, plot_path)

    print("=== VERGLEICH ===")
    print(f"Ergebnis-CSV: {csv_path}")
    print(f"Vergleichsgrafik: {plot_path}")
    for row in rows:
        print(
            f"{row['model']:>3}: mAP50-95={row['mAP50-95']:.4f}, "
            f"mAP50={row['mAP50']:.4f}, P={row['precision']:.4f}, "
            f"R={row['recall']:.4f}"
        )

    if len(rows) == 2:
        delta = float(rows[0]["mAP50-95"]) - float(rows[1]["mAP50-95"])
        print(f"Delta mAP50-95 (new - old): {delta:+.4f}")


def save_comparison_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save a side-by-side comparison of overall and per-class metrics."""
    model_labels = {
        "new": "Neues Modell",
        "old": "Altes Modell",
    }
    colors = {"new": "#167D9A", "old": "#A7B0B8"}
    panels = [
        (
            "Gesamtmetriken",
            ["mAP50-95", "mAP50", "precision", "recall"],
            ["mAP50–95", "mAP50", "Precision", "Recall"],
        ),
        (
            "mAP50–95 je Klasse",
            ["mAP50-95 Apple", "mAP50-95 Banana"],
            ["Apple", "Banana"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    bar_width = 0.34
    for axis, (title, keys, tick_labels) in zip(axes, panels):
        positions = list(range(len(keys)))
        for model_index, row in enumerate(rows):
            model_key = str(row["model"])
            offsets = [x + (model_index - (len(rows) - 1) / 2) * bar_width for x in positions]
            values = [float(row[key]) for key in keys]
            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                color=colors.get(model_key, "#167D9A"),
                label=model_labels.get(model_key, model_key),
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

        axis.set_title(title)
        axis.set_xticks(positions, tick_labels)
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    image_count = rows[0].get("images", "?") if rows else "?"
    fig.suptitle(f"Modellvergleich auf dem Testdatensatz ({image_count} Bilder)", fontsize=14)
    axes[0].set_ylabel("Metrik (0–1)")
    axes[0].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
