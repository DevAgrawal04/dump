# Install Captum for Integrated Gradients if not already installed
# !pip install captum

import torch
from captum.attr import IntegratedGradients
import numpy as np

# Function to visualize the token attributions
def visualize_token_attributions(input_text, attributions, tokenizer):
    tokens = tokenizer.tokenize(input_text)  # Tokenize the input text
    attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # Sum attributions across embedding dimensions

    # Normalize attributions for better visualization
    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)

    # Display tokens with their attribution scores
    for token, score in zip(tokens, attributions[:len(tokens)]):  # Align tokens and attributions
        print(f"{token}: {score:.4f}")

# Function to compute integrated gradients
def compute_integrated_gradients(model, tokenizer, input_text, label, max_len=256, baseline_text="[PAD]", n_steps=50):
    model.eval()

    # Tokenize input and baseline text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device).long()  # Ensure the tensor is of type Long
    attention_mask = inputs['attention_mask'].to(device).long()  # Ensure the tensor is of type Long
    
    # Generate a baseline that matches input shape, usually padded zeros or "[PAD]"
    baseline_ids = tokenizer.encode(
        baseline_text, 
        add_special_tokens=True, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    ).to(device).long()  # Ensure the tensor is of type Long

    # Initialize Integrated Gradients object
    ig = IntegratedGradients(model)

    # Define forward function for the model to focus on the CLS token's output
    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the logits directly or any relevant part (e.g., CLS token output)
        return outputs[:, 0]  # Taking the CLS token output for attribution

    # Compute attributions using integrated gradients
    attributions, delta = ig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=label,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # Visualize attributions
    print(f"Integrated Gradients Delta: {delta.item():.4f}")
    visualize_token_attributions(input_text, attributions, tokenizer)

# Example usage: Apply IG to a specific test sample
sample_text = "Replace this with an example text from your dataset."
true_label = 1  # Replace with the correct label for this example

# Call the function with the sample text
compute_integrated_gradients(model, tokenizer, sample_text, true_label)
