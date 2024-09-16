# import mlflow.pytorch

# # Load the model from MLflow
# model_uri = 'runs:/<run_id>/model'  # Replace <run_id> with your specific run ID
# model = mlflow.pytorch.load_model(model_uri)

# # Print the model architecture
# print(model)

# # Count the number of layers
# num_layers = sum(1 for _ in model.modules())
# print(f'Total number of layers: {num_layers}')


# # Check which layers have gradients
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         if param.grad is not None:
#             print(f'Layer: {name} has been affected during training.')
#         else:
#             print(f'Layer: {name} has not been affected (no gradient).')


import torch
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

# Define model, optimizer, and loss
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
bert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model = SciBERTClassifier(bert_model).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([0.55]).to(device))

# Initialize tracking dictionary
updated_layers = {}

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Track parameters with gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                updated_layers[name] = updated_layers.get(name, 0) + 1

        optimizer.step()

# Report updated layers
print("Layers with updates during training:")
for layer, count in updated_layers.items():
    print(f"{layer} was updated {count} times.")
