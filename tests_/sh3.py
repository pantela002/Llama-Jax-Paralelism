import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

os.makedirs("hf_params", exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load official config
model_id = "meta-llama/Meta-Llama-3.1-8B"
hf_config = AutoConfig.from_pretrained(model_id)
hf_config.num_hidden_layers = 1  # 👈 force only 1 layer 
hf_config.tie_word_embeddings = False 
hf_config.torch_dtype = torch.float16  # 👈 specify float16 in config

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. Build model from modified config (randomly initialized)
model = AutoModelForCausalLM.from_config(hf_config).to(dtype=torch.float16, device=device)
model.eval()

with open("hf_params.txt", "w") as f:
    for name, param in model.named_parameters():
        np_param = param.detach().cpu().numpy()
        f.write(f"{name}: shape={np_param.T.shape}, dtype={np_param.dtype}\n")
        f.write(np.array2string(np_param.T, separator=', ', threshold=10, edgeitems=3))
        f.write("\n\n")


# 3. Create dummy inputs
input_ids = torch.ones((1, 8), dtype=torch.long)            # shape: (batch=1, seq=8)
attention_mask = torch.ones_like(input_ids)
position_ids = torch.arange(8).unsqueeze(0)                 # shape: (1, 8)

# 4. Run model forward pass
with torch.no_grad():
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

# 5. Print results
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 6. Output
print("✅ Logits shape (real weights, unsharded):", output.logits.shape)
print(output.logits)

np.save("output_hf_unsharded.npy", output.logits)
