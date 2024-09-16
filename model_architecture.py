import mlflow.pytorch

# Load the model from MLflow
model_uri = 'runs:/<run_id>/model'  # Replace <run_id> with your specific run ID
model = mlflow.pytorch.load_model(model_uri)

# Print the model architecture
print(model)

# Count the number of layers
num_layers = sum(1 for _ in model.modules())
print(f'Total number of layers: {num_layers}')


# Check which layers have gradients
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            print(f'Layer: {name} has been affected during training.')
        else:
            print(f'Layer: {name} has not been affected (no gradient).')
