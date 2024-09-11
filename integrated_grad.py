# Install Captum for Integrated Gradients
!pip install captum

import torch
from captum.attr import IntegratedGradients
import numpy as np

# Define a helper function to visualize the attributions
def visualize_token_attributions(input_text, attributions, tokenizer):
    tokens = tokenizer.tokenize(input_text)
    attributions = attributions.squeeze(0).sum(dim=1).detach().cpu().numpy()
    
    # Normalize attributions for better visualization
    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)

    # Display tokens with their attribution scores
    for token, score in zip(tokens, attributions):
        print(f"{token}: {score:.4f}")

# Function to compute integrated gradients
def compute_integrated_gradients(model, tokenizer, input_text, label, baseline_text="[PAD]", n_steps=50):
    model.eval()

    # Tokenize input and baseline text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_len, truncation=True).to(device)
    baseline_ids = tokenizer.encode(baseline_text, return_tensors='pt', max_length=max_len, truncation=True).to(device)

    # Prepare attention mask
    input_mask = torch.ones_like(input_ids).to(device)
    
    # Initialize Integrated Gradients object
    ig = IntegratedGradients(model)
    
    # Compute attributions using integrated gradients
    attributions, delta = ig.attribute(
        inputs=input_ids, 
        baselines=baseline_ids, 
        target=label,
        additional_forward_args=(input_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Visualize attributions
    print(f"Integrated Gradients Delta: {delta.item():.4f}")
    visualize_token_attributions(input_text, attributions, tokenizer)

# Example usage: Apply IG to a specific test sample
sample_text = "Replace this with an example text from your dataset."
true_label = 1  # Replace with the correct label for this example

compute_integrated_gradients(model, tokenizer, sample_text, true_label)

