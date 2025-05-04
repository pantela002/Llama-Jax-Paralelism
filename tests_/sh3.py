import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 1. Load official config
model_id = "meta-llama/Meta-Llama-3.1-8B"
hf_config = AutoConfig.from_pretrained(model_id)

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. Build model from modified config (randomly initialized)
model = AutoModelForCausalLM.from_config(hf_config)
model.eval()


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
print("✅ HF Logits shape:", output.logits.shape)
print(output)

# 6. Save to file
with open("output_hf_unsharded.txt", "w") as f:
    f.write(str(output.logits))
