# Saving a model

# Save the model
model_path = 'scibert_cls_model'
torch.save(model.state_dict(), model_path + '/pytorch_model.bin')

# Save the tokenizer
tokenizer_path = 'scibert_cls_tokenizer'
tokenizer.save_pretrained(tokenizer_path)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('scibert_cls_tokenizer')

# Re-initialize the model with the architecture
model = SciBERTClassifier(AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')).to(device)

# Load the saved weights into the model
model.load_state_dict(torch.load('scibert_cls_model/pytorch_model.bin'))

# Set the model to evaluation mode
model.eval()
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('scibert_cls_tokenizer')

# Re-initialize the model with the architecture
model = SciBERTClassifier(AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')).to(device)

# Load the saved weights into the model
model.load_state_dict(torch.load('scibert_cls_model/pytorch_model.bin'))

# Set the model to evaluation mode
model.eval()

# Define helper functions
def prepare_data(texts, tokenizer, max_len):
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device)
    }

def predict(texts):
    data = prepare_data(texts, tokenizer, max_len=256)  # Use the same max_len as during training

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

# Prepare the text data from df2
texts = df2['Processed_Description'].tolist()

# Get predictions
predictions = predict(texts)

# Convert predictions to DataFrame for easier manipulation
pred_df = pd.DataFrame(predictions)

# Add predictions and confidence to df2
df2['Scibert_pred'] = pred_df['predicted_label']
df2['Scibert_prob'] = pred_df['confidence']

# Print the updated DataFrame with predictions
print(df2[['Processed_Description', 'Scibert_pred', 'Scibert_prob']])
