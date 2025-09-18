""" yolo_threshold_checker.py """

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from beetle_cropper import BeetleCropper

def evaluate_thresholds(image_dir, thresholds, expected_positive=True):
    """
    Run YOLO beetle detection at different thresholds and measure acceptance rate.

    Args:
        image_dir (str or Path): Directory of validation images.
        thresholds (list of float): Confidence thresholds to evaluate.
        expected_positive (bool): True if images should contain beetles,
            False if images should not contain beetles (for false-positive testing).

    Returns:
        dict: Mapping from threshold (float) to detection rate (float).
    """
    image_dir = Path(image_dir)
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    results = {}
    for t in thresholds:
        cropper = BeetleCropper(threshold=t)
        detected = 0
        for img_file in image_files:
            img = Image.open(img_file).convert("RGB")
            cropped = cropper.crop_beetle(img)
            if cropped is not None:
                detected += 1

        rate = detected / len(image_files) if len(image_files) > 0 else 0
        results[t] = rate
        if expected_positive:
            print(f"Threshold {t:.2f} - Recall: {rate:.3f}")
        else:
            print(f"Threshold {t:.2f} - False Positive Rate: {rate:.3f}")
    return results


if __name__ == "__main__":
    thresholds = np.linspace(0.1, 0.9, 9) # thresholds from 0.1 to 0.9 in 0.1 steps

    print("\nEvaluating on beetle images - Recall")
    recall_results = evaluate_thresholds("dataset", thresholds, expected_positive=True)

    print("\nEvaluating on non-beetle images - False Positive Rate")
    fpr_results = evaluate_thresholds("non_beetle_images", thresholds, expected_positive=False)

    # Plot results: threshold-performance graph (recall vs FPR)
    plt.figure(figsize=(8, 6))
    plt.plot(list(recall_results.keys()), list(recall_results.values()), marker="o", label="Recall (beetles)")
    plt.plot(list(fpr_results.keys()), list(fpr_results.values()), marker="s", label="False Positive Rate (non-beetles)")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Detection Rate")
    plt.title("Threshold Sweep for YOLO Beetle Detector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = Path("yolo_threshold_graph.png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path.resolve()}")
