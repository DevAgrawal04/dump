import mlflow
import mlflow.pytorch

# Start an MLflow run to track your model
with mlflow.start_run(run_name="SciBERT_Classification") as run:
    # Log the model with MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model, 
        artifact_path="scibert_cls_model", 
        registered_model_name="SciBERT_Classification_Model"
    )
    
    # Log the tokenizer separately
    mlflow.log_artifact('scibert_cls_tokenizer', artifact_path='tokenizer')

print("Model and tokenizer saved successfully with MLflow.")


# Load the saved model using MLflow
loaded_model = mlflow.pytorch.load_model("models:/SciBERT_Classification_Model/latest")

# Set the model to evaluation mode
loaded_model.eval()

# Ensure the tokenizer is loaded from the saved artifact path
tokenizer = AutoTokenizer.from_pretrained('scibert_cls_tokenizer')



def predict_with_mlflow_model(texts, model, tokenizer, max_len=256):
    # Prepare data
    data = prepare_data(texts, tokenizer, max_len)

    # Make predictions
    with torch.no_grad():
        logits = model(data['input_ids'], data['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
    
    results = []
    for pred, prob in zip(preds, probs):
        results.append({
            'predicted_label': 'MI' if pred == 1 else 'Non-MI',
            'confidence': prob if pred == 1 else 1 - prob
        })
    
    return results

# Use the loaded model to make predictions on df2
texts = df2['Processed_Description'].tolist()
predictions = predict_with_mlflow_model(texts, loaded_model, tokenizer)

# Convert predictions to DataFrame and add to df2
pred_df = pd.DataFrame(predictions)
df2['Scibert_pred'] = pred_df['predicted_label']
df2['Scibert_prob'] = pred_df['confidence']

print(df2[['Processed_Description', 'Scibert_pred', 'Scibert_prob']])




### ---------------------------------------------------------------
#mlflow CPU
import mlflow.pytorch
import torch
from transformers import AutoTokenizer

# Set the device to CPU to avoid GPU memory issues
device = torch.device('cpu')

# Load the model using MLflow
model_uri = f"models:/<model_name>/<model_version>"  # Update <model_name> and <model_version> as per your setup
model = mlflow.pytorch.load_model(model_uri, map_location=device)

# Load the tokenizer used during training
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Set the model to evaluation mode
model.eval()

# Function to predict using the loaded model
def predict(texts):
    predictions = []
    probabilities = []

    # Process each text in the input list
    for text in texts:
        # Tokenize and encode the input text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move tensors to CPU
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()  # For binary classification, sigmoid gives the probability

        # Append predictions and probabilities
        pred = 1 if probs[0][0] > 0.5 else 0
        predictions.append(pred)
        probabilities.append(probs[0][0])

    return predictions, probabilities

# Example predictions
texts = df2['Processed_description'].tolist()  # Replace with your DataFrame's column
df2['SciBERT_pred'], df2['SciBERT_prob'] = predict(texts)

# Display the results
print(df2[['Processed_description', 'SciBERT_pred', 'SciBERT_prob']].head())
