# Hugging Face tokenizer
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Your custom tokenizer
from jax_llama.llama3_tokenizer import Tokenizer as CustomTokenizer
custom_tokenizer = CustomTokenizer("/root/tt/sw/llama3.1-8B/original/tokenizer.model")

text = "Q: How many eggs are left? <|end_of_text|> A: 5 eggs <|end_of_text|>"

niz1 = hf_tokenizer.encode(text, add_special_tokens=True)
niz2 = custom_tokenizer.encode(text, bos=True, eos=True)
print("HF IDs:", hf_tokenizer.encode(text, add_special_tokens=True))
print("Custom IDs:", custom_tokenizer.encode(text, bos=True, eos=True, allowed_special="all", disallowed_special=()))

print("HF Decoded:", hf_tokenizer.decode(niz1))
print("Custom Decoded:", custom_tokenizer.decode(niz2))