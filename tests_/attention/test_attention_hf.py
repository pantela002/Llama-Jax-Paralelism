import torch
from transformers import LlamaTokenizerFast, LlamaForCausalLM

def test_attention_only():
    model_id = "meta-llama/Llama-3.1-8B"
    model = LlamaForCausalLM.from_pretrained(model_id)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
    model.eval()

    text = "Example input"
    input_ids = tokenizer.encode(text, return_tensors="pt")  # [1, seq_len]
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        # Step 1: Get embeddings from input tokens
        embeddings = model.model.embed_tokens(input_ids)  # [1, seq_len, hidden_dim]

        # Step 2: Prepare position_ids (needed for rotary embeddings)
        position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]

        # Step 3: Get the attention layer (e.g. first decoder block)
        attn_layer = model.model.layers[0].self_attn

        # Step 4: Forward pass through attention layer
        attn_output, _ = attn_layer(
            hidden_states=embeddings,
            position_ids=position_ids,
            attention_mask=None,        # optional
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

        print("Output shape:", attn_output.shape)
        print("First token vector (truncated):", attn_output[0, 0, :10])

        # Save output to file
        with open("attention_output_hf.txt", "w") as f:
            for token_emb in attn_output.squeeze(0):
                line = " ".join(f"{val.item():.6f}" for val in token_emb)
                f.write(line + "\n")

if __name__ == "__main__":
    test_attention_only()
