""" mc_dropout_threshold_checker.py """

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from beetle_cropper import BeetleCropper
from training_database_creator import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
import globals

def evaluate_thresholds(trainer, view, thresholds, n_samples=30, batch_size=32, title_prefix=""):
    """
    Run MC Dropout uncertainty evaluation at different thresholds and save a plot.

    Args:
        trainer: model trainer with evaluate_uncertainty method.
        view (str): dataset view (e.g. "late", "caud").
        thresholds (list of float): uncertainty thresholds to evaluate.
        n_samples (int): number of MC dropout passes.
        batch_size (int): batch size for evaluation.
        title_prefix (str): extra label for distinguishing plots (e.g. "Species" or "Genus").

    Returns:
        dict: Mapping from threshold -> metrics (coverage, f1, acc).
    """
    base_results = trainer.evaluate_uncertainty(
        view, n_samples=n_samples, batch_size=batch_size, threshold=None
    )

    preds = np.array(base_results["all_preds"])
    labels = np.array(base_results["all_labels"])
    uncertainties = np.array(base_results["all_uncertainties"])

    results = []
    results_dict = {}
    for t in thresholds:
        mask = uncertainties < t
        if mask.sum() == 0:
            f1, acc, coverage = 0.0, 0.0, 0.0
        else:
            f1 = f1_score(labels[mask], preds[mask], average="macro")
            acc = accuracy_score(labels[mask], preds[mask])
            coverage = mask.mean()

        results_dict[t] = {"f1": f1, "accuracy": acc, "coverage": coverage}

        results.append({
            "threshold": t,
            "f1": f1,
            "accuracy": acc,
            "coverage": coverage
        })

        print(f"Threshold {t:.2f} - F1: {f1:.3f}, Acc: {acc:.3f}, Coverage: {coverage:.2f}")
    
    # --- Save CSV ---
    csv_dir = Path("mc_dropout_results")
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"mc_dropout_threshold_results_{title_prefix.lower()}_{view}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path.resolve()}")

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(list(results_dict.keys()), [v["f1"] for v in results_dict.values()],
             marker="o", label="Macro F1")
    plt.plot(list(results_dict.keys()), [v["accuracy"] for v in results_dict.values()],
             marker="s", label="Accuracy")
    plt.plot(list(results_dict.keys()), [v["coverage"] for v in results_dict.values()],
             marker="^", label="Coverage")

    plt.xlabel("Uncertainty Threshold")
    plt.ylabel("Score / Fraction")
    plt.title(f"{title_prefix} MC Dropout Threshold Sweep ({view.upper()} view)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = Path(f"mc_dropout_graphs/mc_dropout_threshold_graph_{title_prefix.lower()}_{view}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path.resolve()}")
    plt.close()

    return results_dict

def evaluate_mc_predictions(trainer, view, n_samples=30, batch_size=32, title_prefix=""):
    """
    Runs MC Dropout Prediction Calculations and then saves and plots data.

    Args:
        trainer: model trainer with evaluate_uncertainty method.
        view (str): dataset view (e.g. "late", "caud").
        n_samples (int): number of MC dropout passes.
        batch_size (int): batch size for evaluation.
        title_prefix (str): extra label for distinguishing plots (e.g. "Species" or "Genus").

    Returns:
        dict: metric correlations (confidence, inverse_entropy, combined_certainty)
    """
    base_results = trainer.evaluate_uncertainty(
        view, n_samples=n_samples, batch_size=batch_size, threshold=None
    )

    preds = np.array(base_results["all_preds"])
    labels = np.array(base_results["all_labels"])
    confidences = np.array(base_results["all_confidences"])
    uncertainties = np.array(base_results["all_uncertainties"])

    # Additional metrics (correctness, certainty)
    correctness = (preds == labels).astype(int)
    # Compute combined certainty score: certainty = confidence × (1 - entropy)
    certainty_score = confidences * (1 - uncertainties)

    # CSV Save
    output_dir = Path("mc_dropout_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "labels": labels,
        "predicted_class": preds,
        "correct": correctness,
        "confidence": confidences,
        "entropy": uncertainties,
        "certainty_score": certainty_score
    })

    sample_csv_path = output_dir / f"mc_dropout_predictions_{title_prefix.lower()}_{view}.csv"
    df.to_csv(sample_csv_path, index=False)
    print(f"Per-sample MC-Dropout CSV saved to {sample_csv_path.resolve()}")

    # Compute correlations with correctness
    conf_corr = np.corrcoef(confidences, correctness)[0, 1]
    inverse_entropy = 1 - uncertainties
    ent_corr = np.corrcoef(inverse_entropy, correctness)[0, 1]  # lower entropy = more correct
    cert_corr = np.corrcoef(certainty_score, correctness)[0, 1]

    print("\nCorrelations:")
    print(f"Confidence vs Correctness:        {conf_corr:.4f}")
    print(f"Inverse Entropy vs Correctness:   {ent_corr:.4f}")
    print(f"Combined Certainty vs Correctness:{cert_corr:.4f}")

    # Plot Accuracy vs Metric Curves
    metrics = {
        "confidence": confidences,
        "inverse_entropy": 1 - uncertainties,
        "combined_certainty": certainty_score
    }

    plt.figure(figsize=(10, 6))
    for name, metric in metrics.items():
        sort_idx = np.argsort(metric)
        sorted_metric = metric[sort_idx]
        sorted_correct = correctness[sort_idx]

        cumulative_acc = []
        for i in range(1, len(metric) + 1):
            cumulative_acc.append(sorted_correct[:i].mean())
        x, y = sorted_metric, cumulative_acc
        plt.plot(x, y, label=name)

    plt.title(f"Accuracy vs Metric – {title_prefix} – {view}")
    plt.xlabel("Metric Value (sorted)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    acc_curve_path = output_dir / f"accuracy_vs_metric_{title_prefix.lower()}_{view}.png"
    plt.savefig(acc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Accuracy-vs-Metric plot saved to {acc_curve_path.resolve()}")

    return {
        "confidence": conf_corr,
        "inverse_entropy": ent_corr,
        "combined_certainty": cert_corr,
    }

if __name__ == "__main__":
    while True:
        print("\nWould you like to use class balancing techniques while training?")
        print("\t0 = No Balancing\n" \
            "\t1 = Class-Weighted Loss Function\n" \
            "\t2 = Oversampling Only\n" \
            "\t3 = Both (Oversampling + Class-Weighted Loss)")
        user_input = int(input("Enter the number of your choice: "))
        if user_input == 0:
            balance_classes = 0
            break
        if user_input == 1:
            balance_classes = 1
            break
        if user_input == 2:
            balance_classes = 2
            break
        if user_input == 3:
            balance_classes = 3
            break
        print("Invalid Input. Please enter 0, 1, 2, or 3.")

    while True:
        print("\nWould you like to preform a threshold sweep, calculate MC predictions, or both?")
        print("\t0 = Threshold Sweep\n" \
            "\t1 = Calculate MC Predictions\n" \
            "\t2 = Both")
        user_input = int(input("Enter the number of your choice: "))
        if user_input == 0:
            experiment = 0
            break
        if user_input == 1:
            experiment = 1
            break
        if user_input == 2:
            experiment = 2
            break
        print("Invalid Input. Please enter 0, 1, or 2.")


    # Create the beetle cropper object to be used in dataset creation and image cropping
    beetle_cropper = BeetleCropper()
    # Crop the images in the original dataset so that the image is only the beetle
    beetle_cropper.build(image_dir="dataset", output_dir=globals.cropped_dataset)

    # Set up data converter
    tdc = TrainingDataConverter(globals.cropped_dataset)
    tdc.conversion(globals.training_database)

    # Final cleanup: remove cropped dataset
    beetle_cropper.cleanup(globals.cropped_dataset)

    # Read converted data
    dbr = DatabaseReader(
        database=globals.training_database, class_file_path=globals.class_list, create_other=False)
    df = dbr.get_dataframe()

    # Display how many images we have for each angle
    print("Number of Images for Each Angle in the Original Dataset:")
    print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
    print(f"DORS: {(df['View'] == 'DORS').sum()}")
    print(f"FRON: {(df['View'] == 'FRON').sum()}")
    print(f"LATE: {(df['View'] == 'LATE').sum()}")


    # initialize number of outputs
    SPECIES_OUTPUTS = dbr.get_num_species()
    GENUS_OUTPUTS = dbr.get_num_genus()

    # Run training with dataframe
    species_tp = TrainingProgram(df, "Species", SPECIES_OUTPUTS, augment=True, balance_classes=balance_classes)

    # Training
    threshold_list = np.linspace(0, 1, 101)  # 0.0 to 1.0 in 0.01 steps
    all_results = {}
    # Species CAUD
    erasure_params_caud = {
        "p": 0.5450068594306283,
        "min": 0.032231275920186486,
        "max": 0.23975077356392424
    }
    species_tp.train_resnet_model(20, "caud", batch=16, rotation=6,
                                brightness=0.0672682540489113, weight_decay=0.1, lrate=0.0002205207835665262,
                                erasure_params=erasure_params_caud, max_os_ratio=1.5)

    # Species CAUD MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for species CAUD view...")
        all_results["species_caud"] = evaluate_thresholds(
            species_tp, view="caud", thresholds=threshold_list, n_samples=20, batch_size=16, title_prefix="Species"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for CAUD view...")
        _ = evaluate_mc_predictions(
            species_tp, view="caud", n_samples=20, batch_size=16, title_prefix="Species"
        )


    # Species DORS
    erasure_params_dors = {
        "p": 0.7757711509313643,
        "min": 0.01008374654178916,
        "max": 0.38794012670750844
    }
    species_tp.train_resnet_model(20, "dors", batch=16, rotation=12,
                                brightness=0.22216817398095146, weight_decay=0.1, lrate=0.0001296278789334687,
                                erasure_params=erasure_params_dors, max_os_ratio=2.5)

    # Species DORS MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for species DORS view...")
        all_results["species_dors"] = evaluate_thresholds(
            species_tp, view="dors", thresholds=threshold_list, n_samples=20, batch_size=16, title_prefix="Species"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for DORS view...")
        _ = evaluate_mc_predictions(
            species_tp, view="dors", n_samples=20, batch_size=16, title_prefix="Species"
        )

    # Species FRON
    erasure_params_fron = {
        "p": 0.14786083200104405,
        "min": 0.08542272176573411,
        "max": 0.3766890143419105
    }
    species_tp.train_resnet_model(20, "fron", batch=16, rotation=7,
                                brightness=0.16052298566019538, weight_decay=0.1, lrate=0.00018151090290770348,
                                erasure_params=erasure_params_fron, max_os_ratio=4.0)

    # Species FRON MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for species FRON view...")
        all_results["species_fron"] = evaluate_thresholds(
            species_tp, view="fron", thresholds=threshold_list, n_samples=20, batch_size=16, title_prefix="Species"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for FRON view...")
        _ = evaluate_mc_predictions(
            species_tp, view="fron", n_samples=20, batch_size=16, title_prefix="Species"
        )

    # Species LATE
    erasure_params_late = {
        "p": 0.005799105801707227,
        "min": 0.08818090418966613,
        "max": 0.2566152645216
    }
    species_tp.train_resnet_model(20, "late", batch=64, rotation=6,
                                brightness=0.29977566775503983, weight_decay=0.1, lrate=0.00012089084719947084,
                                erasure_params=erasure_params_late, max_os_ratio=3.5)

    # Species LATE MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for species LATE view...")
        all_results["species_late"] = evaluate_thresholds(
            species_tp, view="late", thresholds=threshold_list, n_samples=20, batch_size=64, title_prefix="Species"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for LATE view...")
        _ = evaluate_mc_predictions(
            species_tp, view="late", n_samples=20, batch_size=16, title_prefix="Species"
        )

    # Run training with dataframe
    genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS, augment=True, balance_classes=balance_classes)

    # Training
    # Genus CAUD
    erasure_params_caud = {
        "p": 0.117534992000064,
        "min": 0.08054270560117567,
        "max": 0.2983577819330524
    }
    genus_tp.train_resnet_model(20, "caud", batch=16, rotation=10,
                            brightness=0.1462847736327197, weight_decay=0.1, lrate=0.00004409398823911199,
                            erasure_params=erasure_params_caud, max_os_ratio=5.0)
    # Genus CAUD MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus CAUD view...")
        all_results["genus_caud"] = evaluate_thresholds(
            genus_tp, view="caud", thresholds=threshold_list, n_samples=20, batch_size=16, title_prefix="Genus"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for CAUD view...")
        _ = evaluate_mc_predictions(
            genus_tp, view="caud", n_samples=20, batch_size=16, title_prefix="Genus"
        )

    # Genus DORS
    erasure_params_dors = {
        "p": 0.6279748323341047,
        "min": 0.041921505805665914,
        "max": 0.24388226488220693
    }
    genus_tp.train_resnet_model(20, "dors", batch=32, rotation=6,
                            brightness=0.2988104061389692, weight_decay=0.1, lrate=0.00004736821824349854,
                            erasure_params=erasure_params_dors, max_os_ratio=1.0)

    # Genus DORS MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus DORS view...")
        all_results["genus_dors"] = evaluate_thresholds(
            genus_tp, view="dors", thresholds=threshold_list, n_samples=20, batch_size=32, title_prefix="Genus"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for DORS view...")
        _ = evaluate_mc_predictions(
            genus_tp, view="dors", n_samples=20, batch_size=16, title_prefix="Genus"
        )

    # Genus FRON
    erasure_params_fron = {
        "p": 0.30518586009082976,
        "min": 0.04609315007975057,
        "max": 0.36140797065499464
    }
    genus_tp.train_resnet_model(20, "fron", batch=64, rotation=14,
                            brightness=0.22903306674663448, weight_decay=0.1, lrate=0.0001380146193447115,
                            erasure_params=erasure_params_fron, max_os_ratio=5.0)

    # Genus FRON MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus FRON view...")
        all_results["genus_fron"] = evaluate_thresholds(
            genus_tp, view="fron", thresholds=threshold_list, n_samples=20, batch_size=64, title_prefix="Genus"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for FRON view...")
        _ = evaluate_mc_predictions(
            genus_tp, view="fron", n_samples=20, batch_size=16, title_prefix="Genus"
        )

    # Genus LATE
    erasure_params_late = {
        "p": 0.30535724516213314,
        "min": 0.011359991265195598,
        "max": 0.31162030351760406
    }
    genus_tp.train_resnet_model(20, "late", batch=32, rotation=10,
                            brightness=0.04304050259182124, weight_decay=0.1, lrate=0.00001826137626671228,
                            erasure_params=erasure_params_late, max_os_ratio=3.0)

    # Genus LATE MC Dropout
    if experiment == 0 or experiment == 2:
        print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus LATE view...")
        all_results["genus_late"] = evaluate_thresholds(
            genus_tp, view="late", thresholds=threshold_list, n_samples=20, batch_size=32, title_prefix="Genus"
        )
    if experiment == 1 or experiment == 2:
        print("\nRunning Monte Carlo Dropout prediction calculation for LATE view...")
        _ = evaluate_mc_predictions(
            genus_tp, view="late", n_samples=20, batch_size=16, title_prefix="Genus"
        )
