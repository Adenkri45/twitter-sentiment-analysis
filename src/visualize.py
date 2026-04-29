import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Bar Chart for Accuracy and F1-score
def plot_accuracy_f1(models, accuracies, f1_scores):
    # Accuracy Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

    # F1-Score Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(models, f1_scores, color='lightgreen')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

# Confusion Matrix Heatmap
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()
