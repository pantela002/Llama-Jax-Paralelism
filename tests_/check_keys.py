import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os
from jax_llama.convert_weights import convert_state_dict_keys  # your function

# === Setup paths ===
ckpt_dir = "/root/tt/sw/llama3.1-8B/8B"
model_id = "meta-llama/Meta-Llama-3.1-8B"

# 1. Load config
hf_config = AutoConfig.from_pretrained(model_id)
hf_config.num_hidden_layers = 1
hf_config.tie_word_embeddings = False
hf_config.torch_dtype = torch.float16

# 2. Load tokenizer (optional)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load checkpoint
ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
state_dict = torch.load(ckpt_path, map_location="cpu")

# 4. Convert to HF-compatible key names
converted_state_dict = convert_state_dict_keys(state_dict)
converted_state_dict = {
    k: v for k, v in converted_state_dict.items()
    if k.startswith("model.layers.0.") or not k.startswith("model.layers.")
}
# 5. Load model and compare keys
model = AutoModelForCausalLM.from_config(hf_config)

# 6. Compare keys
model_keys = set(model.state_dict().keys())
converted_keys = set(converted_state_dict.keys())

missing_keys = model_keys - converted_keys
extra_keys = converted_keys - model_keys

print("🧩 Missing keys in converted_state_dict (expected by model):")
for k in sorted(missing_keys):
    print("  ", k)

print("\n🔁 Extra keys in converted_state_dict (not used by model):")
for k in sorted(extra_keys):
    print("  ", k)

print(f"\n✅ Total model keys: {len(model_keys)}")
print(f"✅ Total converted keys: {len(converted_keys)}")
