# hf_llama_embedding.py
import torch
from transformers import LlamaModel, LlamaTokenizerFast

def test_embedding():
    model_id = "meta-llama/Llama-3.1-8B"
    model = LlamaModel.from_pretrained(model_id)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)

    text = "Capital of France is"
    # return_tensors="pt" gives a PyTorch tensor
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Extract embeddings
    with torch.no_grad():
        model.eval()
        # Hugging Face stores embeddings in model.get_input_embeddings()
        embeddings_hf = model.get_input_embeddings()(input_ids).squeeze(0)  # [seq_len, dim]

    # Save to text file
    with open("embedding_output_hf.txt", "w") as f:
        for token_emb in embeddings_hf:
            line = " ".join(f"{val.item():.6f}" for val in token_emb)
            f.write(line + "\n")

    print("HF embeddings saved to embedding_output_hf.txt")


if __name__ == "__main__":
    test_embedding()
