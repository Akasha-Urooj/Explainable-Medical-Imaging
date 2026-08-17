from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. Output Directory
# ==========================================

RESULTS_DIR = Path("results")

FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. Experiment Results
# ==========================================

baseline = {
    "Accuracy": 0.9695,
    "Precision": 0.9746,
    "Recall": 0.9846,
    "F1 Score": 0.9795
}

resnet_validation = {
    "Accuracy": 0.9027,
    "Precision": 0.0,   # Not reported in final epoch output
    "Recall": 0.0,      # Not reported in final epoch output
    "F1 Score": 0.9366,
    "ROC-AUC": 0.9733
}

resnet_test = {
    "Accuracy": 0.7692,
    "Precision": 0.7384,
    "Recall": 0.9769,
    "F1 Score": 0.8411,
    "ROC-AUC": 0.9336
}


# ==========================================
# 3. Error Analysis
# ==========================================

false_positives = 135
false_negatives = 9

true_negatives = 99
true_positives = 381


# ==========================================
# 4. Print Research Summary
# ==========================================

print("\n")
print("=" * 60)
print("EXPLAINABLE MEDICAL IMAGING")
print("FINAL RESEARCH RESULTS")
print("=" * 60)


print("\n--- Baseline CNN ---")

for metric, value in baseline.items():

    print(
        f"{metric}: {value:.4f}"
    )


print("\n--- ResNet-18 Validation ---")

for metric, value in resnet_validation.items():

    if value != 0:

        print(
            f"{metric}: {value:.4f}"
        )


print("\n--- ResNet-18 Test Set ---")

for metric, value in resnet_test.items():

    print(
        f"{metric}: {value:.4f}"
    )


print("\n--- Error Analysis ---")

print(
    f"True Negatives : {true_negatives}"
)

print(
    f"False Positives: {false_positives}"
)

print(
    f"False Negatives: {false_negatives}"
)

print(
    f"True Positives  : {true_positives}"
)


# ==========================================
# 5. Comparison Chart
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

baseline_values = [
    baseline["Accuracy"],
    baseline["Precision"],
    baseline["Recall"],
    baseline["F1 Score"]
]

test_values = [
    resnet_test["Accuracy"],
    resnet_test["Precision"],
    resnet_test["Recall"],
    resnet_test["F1 Score"]
]


x = np.arange(
    len(metrics)
)

width = 0.35


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline CNN"
)

plt.bar(
    x + width / 2,
    test_values,
    width,
    label="ResNet-18 Test"
)

plt.xticks(
    x,
    metrics
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.title(
    "Baseline CNN vs ResNet-18"
)

plt.legend()

plt.tight_layout()


comparison_path = (
    FIGURES_DIR /
    "final_model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 6. Error Distribution Chart
# ==========================================

error_labels = [
    "False Positives",
    "False Negatives"
]

error_values = [
    false_positives,
    false_negatives
]


plt.figure(
    figsize=(8, 6)
)

plt.bar(
    error_labels,
    error_values
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "ResNet-18 Test Set Errors"
)

plt.tight_layout()


error_chart_path = (
    FIGURES_DIR /
    "error_distribution.png"
)

plt.savefig(
    error_chart_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 7. Save Text Report
# ==========================================

report_path = (
    RESULTS_DIR /
    "final_results_summary.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        "EXPLAINABLE MEDICAL IMAGING\n"
    )

    file.write(
        "FINAL RESEARCH RESULTS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )


    file.write(
        "BASELINE CNN\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    for metric, value in baseline.items():

        file.write(
            f"{metric}: {value:.4f}\n"
        )


    file.write(
        "\nRESNET-18 VALIDATION\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    for metric, value in resnet_validation.items():

        if value != 0:

            file.write(
                f"{metric}: {value:.4f}\n"
            )


    file.write(
        "\nRESNET-18 TEST SET\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    for metric, value in resnet_test.items():

        file.write(
            f"{metric}: {value:.4f}\n"
        )


    file.write(
        "\nERROR ANALYSIS\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        f"True Negatives: {true_negatives}\n"
    )

    file.write(
        f"False Positives: {false_positives}\n"
    )

    file.write(
        f"False Negatives: {false_negatives}\n"
    )

    file.write(
        f"True Positives: {true_positives}\n"
    )


    file.write(
        "\nEXPLAINABILITY\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        "Grad-CAM generated for sample test images.\n"
    )

    file.write(
        "Grad-CAM generated for false-negative cases.\n"
    )

    file.write(
        "Grad-CAM generated for false-positive cases.\n"
    )


# ==========================================
# 8. Completion
# ==========================================

print("\n")
print("=" * 60)

print(
    "Final results summary created successfully!"
)

print(
    f"Comparison chart: {comparison_path}"
)

print(
    f"Error chart: {error_chart_path}"
)

print(
    f"Research summary: {report_path}"
)

print("=" * 60)