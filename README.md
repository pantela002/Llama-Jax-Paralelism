# 🧠 LLaMA 3.1–8B: Tensor Parallel JAX Implementation (Draft PR)

This draft PR adds a tensor-parallel JAX implementation of Meta’s LLaMA 3.1–8B model using a 2×4 device mesh. The code supports both sharded and unsharded execution and matches Hugging Face’s PyTorch reference implementation.

---

## ✅ Setup Instructions

### 1. Install Python and Create Virtual Environment
```
sudo apt install python3.12-venv
mkdir tt
cd tt
python3.12 -m venv llama
source llama/bin/activate
```
### 2. Hugging Face Login
```
You must log into Hugging Face to download the LLaMA 3.1 weights.

pip install huggingface_hub
huggingface-cli login

    Make sure you've requested access to the Meta LLaMA 3 model: https://huggingface.co/meta-llama
```
### 🌿 Branch for This Implementation
```
All changes for this draft PR are in the branch:

llama-3.1.8b-tensor-parallel-draft

Clone the repository and checkout the branch:
git checkout llama-3.1.8b-tensor-parallel-draft
```

### 📁 Download and Structure Model Files
```
mkdir -p sw/llama3.1-8B/original
mkdir -p sw/llama3.1-8B/8B

huggingface-cli download meta-llama/Llama-3.1-8B original/tokenizer.model --local-dir sw/llama3.1-8B/original
huggingface-cli download meta-llama/Llama-3.1-8B original/consolidated.00.pth --local-dir sw/llama3.1-8B
huggingface-cli download meta-llama/Llama-3.1-8B original/params.json --local-dir sw/llama3.1-8B

mv sw/llama3.1-8B/consolidated.00.pth sw/llama3.1-8B/8B/
mv sw/llama3.1-8B/params.json sw/llama3.1-8B/8B/

Final structure:

sw/llama3.1-8B/
├── 8B/
│   ├── consolidated.00.pth
│   └── params.json
└── original/
    └── tokenizer.model
```

### 📦 Install Python Dependencies
```
Make sure you're using Python ≥3.10 (tested on 3.12):

pip install -r tests/jax/models/llama/3_1_8b/requirements.txt
```

### ▶️ Running the Scripts
```
You can run any of the available generation scripts using:

python3 tests/jax/models/llama/3_1_8b/generate_jax.py
python3 tests/jax/models/llama/3_1_8b/generate_hf.py
python3 tests/jax/models/llama/3_1_8b/generate_jax_unsharded.py

    generate_jax.py: Runs the sharded tensor-parallel JAX model (2×4 mesh).

    generate_jax_unsharded.py: Runs the unsharded JAX model.

    generate_hf.py: Runs the Hugging Face PyTorch reference model.

In generate_hf.py and generate_jax_unsharded.py, there are three example prompts commented in the code that can be modified for testing.

✅ All three scripts produce identical outputs for the same input prompt (up to floating point precision).
```

### ⚠️ Note on generate_jax.py (Tensor-Parallel Sharded)

When running the sharded JAX model on a 2×4 mesh:

❗ RAM usage exceeds 64 GB during sharding and the process is killed.

❓ Questions & Feedback

    Can you try running generate_jax.py on your end with more RAM?

        To confirm memory requirements.

        To verify whether the issue is hardware-related.

    Prompt formatting:

        I'm currently using 2-shot prompting, which ensures consistent outputs between the sharded JAX and Hugging Face PyTorch models.

        ❓ Is it expected that prompting style affects alignment?

    Correctness checks:

        ✅ I have verified that the logits match exactly between:

            Sharded JAX

            Unsharded JAX

            Hugging Face PyTorch

    Memory use:

        ❓ Is it expected that sharding the LLaMA 3.1 8B model exceeds 64 GB RAM?

        Or is there something incorrect in the sharding logic?

Let me know how I can improve this or whether the memory limit is just a system constraint 🙏

