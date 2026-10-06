# THIS FILE IS WRITTEN BY AI


"""Compare the Fruit-360 and own-data classifiers on the held-out test set."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "fruit_360": ROOT
    / "models"
    / "trained"
    / "Classification_1_Fruit_360"
    / "weights"
    / "best.pt",
    "own_data": ROOT
    / "models"
    / "trained"
    / "Classification_1_own_data"
    / "weights"
    / "best.pt",
    "fruit_360_plus_own": ROOT
    / "models"
    / "trained"
    / "Classification_1_Fruit_360_plus_own_Model"
    / "weights"
    / "best.pt",
}

MODEL_LABELS = {
    "fruit_360": "Fruit 360",
    "own_data": "Eigene Daten",
    "fruit_360_plus_own": "Fruit 360 + eigene Daten",
}
CLASS_NAMES = {0: "Apple", 1: "Banana"}
DATASET_DIR = ROOT / "trainingsdata" / "classification" / "Eigener Datensatz"
TEST_DIR = DATASET_DIR / "test"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Both models are evaluated at the same classifier input size for a fair comparison.
IMAGE_SIZE = 224


def get_test_image_counts() -> dict[str, int]:
    """Validate the folder-classification test split and count its images."""
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Testordner nicht gefunden: {TEST_DIR}")

    expected_classes = {name.lower() for name in CLASS_NAMES.values()}
    class_dirs = [path for path in TEST_DIR.iterdir() if path.is_dir()]
    actual_classes = {path.name.lower() for path in class_dirs}
    if actual_classes != expected_classes:
        raise ValueError(
            f"Testklassen unter {TEST_DIR} sind {sorted(actual_classes)}; "
            f"erwartet werden {sorted(expected_classes)}."
        )

    counts: dict[str, int] = {}
    for class_id, class_name in CLASS_NAMES.items():
        class_dir = next(path for path in class_dirs if path.name.lower() == class_name.lower())
        images = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            raise FileNotFoundError(f"Keine Testbilder gefunden in: {class_dir}")
        counts[class_name] = len(images)

    return counts


def check_model_classes(model: YOLO, weight_path: Path) -> None:
    """Ensure a classifier uses the same ordered classes as the dataset."""
    if model.task != "classify":
        raise ValueError(f"{weight_path.name} ist ein {model.task}-Modell, kein Klassifizierer.")

    names = model.names
    ordered_names = [names[index].strip().lower() for index in sorted(names)]
    expected_names = [CLASS_NAMES[index].lower() for index in sorted(CLASS_NAMES)]
    if ordered_names != expected_names:
        raise ValueError(
            f"Klassen in {weight_path.name} sind {ordered_names}; erwartet wird "
            f"{expected_names} in dieser Reihenfolge."
        )


def metrics_from_confusion_matrix(matrix: np.ndarray) -> dict[str, float]:
    """Calculate accuracy and macro/per-class precision, recall, and F1."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"Unerwartete Größe der Confusion Matrix: {matrix.shape}")

    # Ultralytics stores predicted classes in rows and true classes in columns.
    per_class: dict[str, dict[str, float]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[class_id, :].sum() - true_positive
        false_negative = matrix[:, class_id].sum() - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    total = matrix.sum()
    result = {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "precision_macro": float(np.mean([value["precision"] for value in per_class.values()])),
        "recall_macro": float(np.mean([value["recall"] for value in per_class.values()])),
        "f1_macro": float(np.mean([value["f1"] for value in per_class.values()])),
    }
    for class_name, values in per_class.items():
        for metric_name, value in values.items():
            result[f"{metric_name} {class_name}"] = value
    return result


def main() -> None:
    class_counts = get_test_image_counts()
    total_images = sum(class_counts.values())
    for path in MODEL_PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Modellgewicht nicht gefunden: {path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "runs" / "classification_test_metric"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"comparison_{run_id}.csv"
    rows: list[dict[str, object]] = []

    print(f"Testbilder: {total_images} aus {TEST_DIR}")
    print(
        f"Apple: {class_counts['Apple']} | Banana: {class_counts['Banana']} | "
        f"imgsz: {IMAGE_SIZE}"
    )
    print("Auswertung startet. Beide Modelle erhalten dieselben Bilder und Einstellungen.\n")

    for label, weight_path in MODEL_PATHS.items():
        print(f"=== {MODEL_LABELS[label]}: {weight_path.name} ===")
        model = YOLO(str(weight_path))
        check_model_classes(model, weight_path)
        metrics = model.val(
            data=str(DATASET_DIR),
            split="test",
            imgsz=IMAGE_SIZE,
            plots=True,
            project=str(results_dir),
            name=f"{run_id}_{label}",
            verbose=True,
        )

        confusion_matrix = metrics.confusion_matrix.matrix
        row: dict[str, object] = {
            "model": MODEL_LABELS[label],
            "weights": str(weight_path.relative_to(ROOT)),
            "images": total_images,
            "Apple images": class_counts["Apple"],
            "Banana images": class_counts["Banana"],
            **metrics_from_confusion_matrix(confusion_matrix),
        }
        rows.append(row)

        print(
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro Precision={row['precision_macro']:.4f} | "
            f"Macro Recall={row['recall_macro']:.4f} | "
            f"Macro F1={row['f1_macro']:.4f}\n"
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
            f"{row['model']:>12}: Accuracy={row['accuracy']:.4f}, "
            f"Macro F1={row['f1_macro']:.4f}, "
            f"Apple Recall={row['recall Apple']:.4f}, "
            f"Banana Recall={row['recall Banana']:.4f}"
        )

    if len(rows) > 1:
        print("=== PAARWEISE DIFFERENZEN ===")
        for first_index, first_row in enumerate(rows):
            for second_row in rows[first_index + 1 :]:
                accuracy_delta = float(first_row["accuracy"]) - float(second_row["accuracy"])
                f1_delta = float(first_row["f1_macro"]) - float(second_row["f1_macro"])
                print(
                    f"{first_row['model']} - {second_row['model']}: "
                    f"Accuracy {accuracy_delta:+.4f} | Macro F1 {f1_delta:+.4f}"
                )


def save_comparison_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save overall and per-class comparison charts."""
    colors = {
        "Fruit 360": "#167D9A",
        "Eigene Daten": "#A7B0B8",
        "Fruit 360 + eigene Daten": "#E59B35",
    }
    panels = [
        (
            "Gesamtmetriken",
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"],
        ),
        (
            "Recall je Klasse",
            ["recall Apple", "recall Banana"],
            ["Apple", "Banana"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    bar_width = 0.34
    for axis, (title, keys, tick_labels) in zip(axes, panels):
        positions = list(range(len(keys)))
        for model_index, row in enumerate(rows):
            model_label = str(row["model"])
            offsets = [
                x + (model_index - (len(rows) - 1) / 2) * bar_width
                for x in positions
            ]
            values = [float(row[key]) for key in keys]
            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                color=colors.get(model_label, "#167D9A"),
                label=model_label,
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

        axis.set_title(title)
        axis.set_xticks(positions, tick_labels)
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    image_count = rows[0].get("images", "?") if rows else "?"
    fig.suptitle(f"Klassifikatorvergleich auf dem Testdatensatz ({image_count} Bilder)", fontsize=14)
    axes[0].set_ylabel("Metrik (0–1)")
    axes[0].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
