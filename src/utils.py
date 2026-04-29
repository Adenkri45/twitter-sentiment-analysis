import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def save_and_return_results(y_true, y_pred, model_name="model"):
    os.makedirs("results", exist_ok=True)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = f"results/{model_name}_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"✅ Classification report saved: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = f"results/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")

    return acc
