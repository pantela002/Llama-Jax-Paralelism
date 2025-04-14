# jax_llama_embedding.py
import jax
import jax.numpy as jnp
import equinox as eqx
from transformers import LlamaModel, LlamaTokenizerFast

class LlamaEmbedding(eqx.Module):
    weight: jnp.ndarray
    param_dtype: any
    compute_dtype: any

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        pretrained_weights=None,
        key=None,
    ):
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        
        if key is None:
            key = jax.random.PRNGKey(99)
        
        if pretrained_weights is None:
            # Random init if no pretrained weights
            self.weight = jax.random.normal(
                key, (num_embeddings, embedding_dim), dtype=self.param_dtype
            )
        else:
            # Load pretrained weights
            self.weight = jnp.array(pretrained_weights, dtype=self.param_dtype)

    def __call__(self, x):
        weight = self.weight.astype(self.compute_dtype)
        embeddings = jnp.take(weight, x, axis=0)
        return embeddings.astype(self.compute_dtype)


def main():
    model_id = "meta-llama/Llama-3.1-8B"
    # Load HF model to extract embedding weights
    from transformers import LlamaModel, LlamaTokenizerFast

    model_id = "meta-llama/Llama-3.1-8B"
    hf_model = LlamaModel.from_pretrained(model_id)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
    pretrained_weights = hf_model.get_input_embeddings().weight.data.cpu().numpy()

    
    text = "Example input"
    # return_tensors="np" gives a NumPy array
    input_ids = tokenizer.encode(text, return_tensors="np").squeeze(0)
    input_ids_jax = jnp.array(input_ids)

    # Build our custom embedding layer with HF weights
    num_embeddings, embedding_dim = pretrained_weights.shape
    llama_embedding_layer = LlamaEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        pretrained_weights=pretrained_weights
    )

    # Get JAX embeddings
    embeddings = llama_embedding_layer(input_ids_jax)  # shape: [seq_len, dim]

    # Save to text file
    with open("embedding_output_jax.txt", "w") as f:
        for token_emb in embeddings:
            line = " ".join(f"{val:.6f}" for val in token_emb.tolist())
            f.write(line + "\n")

    print("JAX embeddings saved to embedding_output_jax.txt")


if __name__ == "__main__":
    main()
