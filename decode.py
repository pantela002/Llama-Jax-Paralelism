import sys
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer

def decode_tokens_file():
    tokens_file_path = "/root/tt/hf_llama/merged_tokens_hf.txt"
    tokenizer_path = "/root/tt/3_1_8b/Llama-Jax-Paralelism/llama3.1-8B/original/tokenizer.model"
    # Initialize tokenizer
    tokenizer = LLaMA3Tokenizer(model_path=tokenizer_path)

    with open(tokens_file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        token_ids = list(map(int, line.strip().split()))
        decoded = tokenizer.decode(token_ids)
        print(f"[Sample {i}]")
        print("Token IDs:", token_ids)
        print("Decoded text:", decoded)
        print("=" * 50)

if __name__ == "__main__":

    decode_tokens_file()
