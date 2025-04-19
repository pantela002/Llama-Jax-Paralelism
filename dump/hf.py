
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# Load the tokenizer and model
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set pad token to eos token

model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval() # Set model to evaluation mode

def print_model(model):
    print(model)
    print(model.config)

def output():    
    # Define the prompt
    prompt = "The capital city of France is"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    np.save("hf_input_ids.npy", input_ids.numpy())
    np.save("hf_attention_mask.npy", attention_mask.numpy())

    # Generate the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the index of the last token
    last_token_index = input_ids.shape[-1] - 1

    # Get logits for the next token
    next_token_logits = logits[:, last_token_index, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    # Decode the next token
    next_word = tokenizer.decode(next_token_id)
    print("Next word prediction:", next_word)

#output()
output()