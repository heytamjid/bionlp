import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_predictions(pred_file, gold_file):
    # Load the predictions and gold labels
    with open(pred_file, "r", encoding="utf-8") as f:
        preds_data = json.load(f)

    with open(gold_file, "r", encoding="utf-8") as f:
        golds_data = json.load(f)

    # Create dictionaries to map ID to label for easy alignment
    pred_dict = {item["id"]: item["label"] for item in preds_data}
    gold_dict = {item["id"]: item["label"] for item in golds_data}

    y_true = []
    y_pred = []

    # Align the predictions and gold labels by ID
    for id_val, true_label in gold_dict.items():
        if id_val in pred_dict:
            y_true.append(true_label)
            y_pred.append(pred_dict[id_val])
        else:
            print(f"Warning: {id_val} not found in predictions.")

    # Generate the standard metrics WITHOUT considering the 0 class
    # We pass labels=[1, 2, 3, 4, 5, 6, 7, 8] to the classification_report
    eval_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    report = classification_report(
        y_true, y_pred, labels=eval_labels, digits=6, zero_division=0
    )

    print("=== Classification Report (Excluding Class 0) ===")
    print(report)

    # Generate the confusion matrix for all classes 0-8
    labels_all = list(range(9))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_all,
        yticklabels=labels_all,
    )
    plt.title("Confusion Matrix (Classes 0-8)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save the plot as a PNG image
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")


if __name__ == "__main__":
    evaluate_predictions("prediction3oncoda.json", "test_label.json")
