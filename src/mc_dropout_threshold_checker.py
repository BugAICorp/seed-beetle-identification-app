""" mc_dropout_threshold_checker.py """

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from beetle_cropper import BeetleCropper
from training_data_converter import TrainingDataConverter
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

    results = {}
    for t in thresholds:
        mask = uncertainties < t
        if mask.sum() == 0:
            f1, acc, coverage = 0.0, 0.0, 0.0
        else:
            f1 = f1_score(labels[mask], preds[mask], average="macro")
            acc = accuracy_score(labels[mask], preds[mask])
            coverage = mask.mean()

        results[t] = {"f1": f1, "accuracy": acc, "coverage": coverage}
        print(f"Threshold {t:.2f} - F1: {f1:.3f}, Acc: {acc:.3f}, Coverage: {coverage:.2f}")

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(list(results.keys()), [v["f1"] for v in results.values()],
             marker="o", label="Macro F1")
    plt.plot(list(results.keys()), [v["accuracy"] for v in results.values()],
             marker="s", label="Accuracy")
    plt.plot(list(results.keys()), [v["coverage"] for v in results.values()],
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

    return results

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
    thresholds = np.linspace(0.0, 1.0, 11)  # 0.0 to 1.0 in 0.1 steps
    # Species CAUD
    erasure_params_caud = {
        "p": 0.3978357251429255,
        "min": 0.04237603082954706,
        "max": 0.3025963685284483
    }
    species_tp.train_resnet_model(
        20, "caud", batch=16, rotation=1, brightness=0.0052372665532581155, lrate=0.000208665948737891,
        erasure_params=erasure_params_caud)

    # Species CAUD MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for species CAUD view...")
    results = evaluate_thresholds(
        species_tp, view="caud", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Species"
    )

    # Species DORS
    erasure_params_dors = {
        "p": 0.5763301129483613,
        "min": 0.06044662804540117,
        "max": 0.18387577071515754
    }
    species_tp.train_resnet_model(
        20, "dors", batch=16, rotation=8, brightness=0.11266599539746057, lrate=0.00016310975593889832,
        erasure_params=erasure_params_dors)

    # Species DORS MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for species DORS view...")
    results = evaluate_thresholds(
        species_tp, view="dors", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Species"
    )

    # Species FRON
    erasure_params_fron = {
        "p": 0.265585095702728,
        "min": 0.071779115882381,
        "max": 0.29234187228616554
    }
    species_tp.train_resnet_model(
        20, "fron", batch=32, rotation=12, brightness=0.14763773752336606, lrate=0.00018738820725043863,
        erasure_params=erasure_params_fron)

    # Species FRON MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for species FRON view...")
    results = evaluate_thresholds(
        species_tp, view="fron", thresholds=thresholds, n_samples=20, batch_size=32, title_prefix="Species"
    )

    # Species LATE
    erasure_params_late = {
        "p": 0.5189325280363017,
        "min": 0.03843699036307908,
        "max": 0.11129682877722781
    }
    species_tp.train_resnet_model(
        20, "late", batch=16, rotation=18, brightness=0.10813954357888121, lrate=0.0001659616690805536,
        erasure_params=erasure_params_late)

    # Species LATE MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for species LATE view...")
    results = evaluate_thresholds(
        species_tp, view="late", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Species"
    )

    # Run training with dataframe
    genus_tp = TrainingProgram(df, "Genus", GENUS_OUTPUTS, augment=True, balance_classes=balance_classes)

    # Training
    # Genus CAUD
    erasure_params_caud = {
        "p": 0.3127187908868738,
        "min": 0.04046194894255532,
        "max": 0.29175754421281885
    }
    genus_tp.train_resnet_model(
        20, "caud", batch=16, rotation=0, brightness=0.2983619077722387, lrate=0.00039800494669978446,
        erasure_params=erasure_params_caud)

    # Genus CAUD MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus CAUD view...")
    results = evaluate_thresholds(
        genus_tp, view="caud", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Genus"
    )

    # Genus DORS
    erasure_params_dors = {
        "p": 0.08429225010786912,
        "min": 0.05881609232667761,
        "max": 0.29034641815208423
    }
    genus_tp.train_resnet_model(
        20, "dors", batch=16, rotation=9, brightness=0.05452949803911396, lrate=0.00034069432228042864,
        erasure_params=erasure_params_dors)

    # Genus DORS MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus DORS view...")
    results = evaluate_thresholds(
        genus_tp, view="dors", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Genus"
    )

    # Genus FRON
    erasure_params_fron = {
        "p": 0.7558902433519469,
        "min": 0.07276752102604624,
        "max": 0.1953562902391759
    }
    genus_tp.train_resnet_model(
        20, "fron", batch=32, rotation=4, brightness=0.17667183838225514, lrate=0.0001997249630754838,
        erasure_params=erasure_params_fron)

    # Genus FRON MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus FRON view...")
    results = evaluate_thresholds(
        genus_tp, view="fron", thresholds=thresholds, n_samples=20, batch_size=32, title_prefix="Genus"
    )

    # Genus LATE
    erasure_params_late = {
        "p": 0.3860968267885073,
        "min": 0.09392431854817945,
        "max": 0.2564630945836204
    }
    genus_tp.train_resnet_model(
        20, "late", batch=16, rotation=10, brightness=0.25458958614413363, lrate=0.00010421711239748923,
        erasure_params=erasure_params_late)

    # Genus LATE MC Dropout
    print("\nRunning Monte Carlo Dropout uncertainty evaluation for genus LATE view...")
    results = evaluate_thresholds(
        genus_tp, view="late", thresholds=thresholds, n_samples=20, batch_size=16, title_prefix="Genus"
    )
