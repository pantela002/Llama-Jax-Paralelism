import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from jax_llama.convert_weights import convert_state_dict_keys  # assumes this exists in your setup

# === Directories ===
ckpt_dir = "/root/tt/sw/llama3.1-8B/8B"
model_id = "meta-llama/Meta-Llama-3.1-8B"
os.makedirs("hf_params", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and override config
hf_config = AutoConfig.from_pretrained(model_id)
hf_config.num_hidden_layers = 1
hf_config.tie_word_embeddings = False
hf_config.torch_dtype = torch.float16

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load checkpoint state_dict
ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])  # pick first for now
state_dict = torch.load(ckpt_path, map_location="cpu")



# 4. Build model and load weights
model = AutoModelForCausalLM.from_config(hf_config).to(dtype=torch.float16, device=device)
converted_state_dict = convert_state_dict_keys(state_dict)
converted_state_dict = {
    k: v for k, v in converted_state_dict.items()
    if k.startswith("model.layers.0.") or not k.startswith("model.layers.")
}

model.load_state_dict(converted_state_dict, strict=True)
model.eval()

"""for name, param in model.named_parameters():
    if name in converted_state_dict:
        ckpt_tensor = converted_state_dict[name].detach().cpu()
        if ckpt_tensor.dtype == torch.bfloat16:
            ckpt_tensor = ckpt_tensor.to(torch.float32)  # convert to float32 for numpy
        model_tensor = param.detach().cpu()
        if model_tensor.dtype == torch.bfloat16:
            model_tensor = model_tensor.to(torch.float32)
        ckpt_np = ckpt_tensor.numpy()
        model_np = model_tensor.numpy()
        diff = np.abs(ckpt_np - model_np).max()
        print(f"{name} max diff: {diff}")
    else:
        print(f"⚠️  {name} not found in converted_state_dict")

os.makedirs("hf_converted_weights_txt", exist_ok=True)
for name, tensor in converted_state_dict.items():
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)  # Convert to float32 for saving

    np_tensor = tensor.numpy()
    txt_path = os.path.join("hf_converted_weights_txt", name.replace(".", "_") + ".txt")

    with open(txt_path, "w") as f:
        f.write(f"{name}: shape={np_tensor.shape}, dtype={np_tensor.dtype}\n")
        f.write(np.array2string(np_tensor, separator=', ', threshold=10, edgeitems=3))
        f.write("\n\n")"""



# 6. Run dummy inference
input_ids = torch.ones((1, 8), dtype=torch.int32).to(device)
attention_mask = torch.ones_like(input_ids)
position_ids = torch.arange(8).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

# 7. Save logits
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print("✅ Logits shape:", output.logits.shape)
print(output.logits)
np.save("output_hf_unsharded.npy", output.logits.cpu().numpy())






