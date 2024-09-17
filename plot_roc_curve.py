from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, loader, dataset_name="Test"):
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()  # True labels

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()  # Predicted probabilities

            y_probs.extend(probs)
            y_true.extend(labels)

    # Calculate ROC curve metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Optionally, return AUC for further analysis
    return roc_auc

roc_auc = plot_roc_curve(model, test_loader, dataset_name="Test")
print(f'Test AUC: {roc_auc:.2f}')
