# Hugging Face tokenizer
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama3.1-8B")

# Your custom tokenizer
from jax_llama.llama3_tokenizer import Tokenizer as CustomTokenizer
custom_tokenizer = CustomTokenizer("/root/tt/sw/llama3.1-8B/original/tokenizer.model")

text = "Q: How many eggs are left?"

print("HF IDs:", hf_tokenizer.encode(text, add_special_tokens=False))
print("Custom IDs:", custom_tokenizer.encode(text, bos=False, eos=False))