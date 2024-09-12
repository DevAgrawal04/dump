# Install necessary packages
!pip install transformers torch scikit-learn mlflow matplotlib

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, accuracy_score, classification_report
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# Enable automatic logging
mlflow.autolog()

# Custom dataset class for the incident descriptions
class IncidentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load the tokenizer and SciBERT model
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
base_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

max_len = 256  # Adjust based on your text length

# Split dataset into train and test sets
X = df2['Processed_Description'].values
y = df2['MI_Incident'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_dataset = IncidentDataset(X_train, y_train, tokenizer, max_len)
test_dataset = IncidentDataset(X_test, y_test, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a classifier based on SciBERT
class SciBERTClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_labels=1):
        super(SciBERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

model = SciBERTClassifier(base_model).to(device)

# Use weighted BinaryCrossEntropy for imbalanced classes
class_weights = torch.tensor([0.55]).to(device)  # Adjust the weights based on class imbalance
criterion = BCEWithLogitsLoss(pos_weight=class_weights)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

# Initialize lists to store loss values
train_losses = []
val_losses = []

# Start an MLflow run
mlflow.start_run(run_name="SciBERT Training")

# Log parameters to MLflow
mlflow.log_param("max_len", max_len)
mlflow.log_param("learning_rate", 2e-5)
mlflow.log_param("epochs", epochs)

# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Log training loss to MLflow
    mlflow.log_metric("avg_training_loss", avg_train_loss, step=epoch)
    print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}')
    
    # Evaluate on validation/test set
    val_loss, val_accuracy = evaluate(model, test_loader, "Test", df2, log_to_mlflow=True)
    val_losses.append(val_loss)

    # Log validation loss and accuracy to MLflow
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

# End the MLflow run
mlflow.end_run()

# Plotting training and validation loss vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, marker='o', linestyle='-', label='Validation Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Define the evaluation function with logging to MLflow
def evaluate(model, loader, dataset_name="Test", df2=None, log_to_mlflow=False):
    model.eval()
    total_loss = 0
    y_preds = []
    y_true = []
    misclassified_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            y_preds.extend(preds)
            y_true.extend(labels)

            # Track misclassified samples with associated probabilities
            for i, (pred, true, prob) in enumerate(zip(preds, labels, probs)):
                if pred != true:
                    global_idx = batch_idx * loader.batch_size + i
                    misclassified_samples.append((global_idx, batch['text'][i], pred, true, prob))

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(y_true, y_preds)
    print(f'{dataset_name} Accuracy: {accuracy * 100:.2f}%')

    # Log evaluation metrics to MLflow
    if log_to_mlflow:
        mlflow.log_metric(f"{dataset_name}_accuracy", accuracy)

    # Print misclassified descriptions with probabilities
    if df2 is not None:
        print(f'\n{dataset_name} Misclassified Samples:')
        for idx, text, pred, true, prob in misclassified_samples:
            if idx < len(df2):
                print(f"Description: {df2['Description'].iloc[idx]}")
                print(f"Text: {text}\nPredicted Label: {pred}, True Label: {true}, Probability: {prob:.4f}\n")

    return avg_loss, accuracy

# Save the trained model and tokenizer
model.bert.save_pretrained('scibert_cls_model')
tokenizer.save_pretrained('scibert_cls_tokenizer')
