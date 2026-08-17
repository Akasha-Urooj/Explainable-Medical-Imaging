import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# Experiment Results
# ==========================================

models = [
    "Baseline CNN",
    "ResNet-18"
]

accuracy = [
    0.9695,
    0.0   # Replace with ResNet accuracy
]

precision = [
    0.9746,
    0.0   # Replace with ResNet precision
]

recall = [
    0.9846,
    0.0   # Replace with ResNet recall
]

f1 = [
    0.9795,
    0.0   # Replace with ResNet F1
]


# ==========================================
# Print Results
# ==========================================

print("\nModel Comparison")
print("=" * 50)

for i, model in enumerate(models):

    print(f"\n{model}")

    print(f"Accuracy : {accuracy[i]:.4f}")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall   : {recall[i]:.4f}")
    print(f"F1 Score : {f1[i]:.4f}")


# ==========================================
# Create Results Directory
# ==========================================

output_dir = Path("results/figures")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# Plot Comparison
# ==========================================

x = range(len(models))

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    accuracy,
    marker="o",
    label="Accuracy"
)

plt.plot(
    x,
    precision,
    marker="o",
    label="Precision"
)

plt.plot(
    x,
    recall,
    marker="o",
    label="Recall"
)

plt.plot(
    x,
    f1,
    marker="o",
    label="F1 Score"
)

plt.xticks(
    list(x),
    models
)

plt.ylim(
    0.8,
    1.0
)

plt.ylabel("Score")

plt.title(
    "Baseline CNN vs ResNet-18"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# ==========================================
# Save Plot
# ==========================================

output_path = (
    output_dir /
    "model_comparison.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nComparison chart saved at: {output_path}"
)
