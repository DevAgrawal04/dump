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
