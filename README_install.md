# Encodec Explorer Environment

Simple setup for the audio processing notebook.

## Installation

**Open Anaconda Prompt (Windows) or Terminal (macOS/Linux) and run:**

```bash
# 1. Create the environment
conda create -n encodec_explore python=3.9 -y

# 2. Activate it
conda activate encodec_explore

# 3. Install packages
pip install -r requirements.txt

# 4. Register Jupyter kernel
python -m ipykernel install --user --name encodec_explore --display-name "Encodec Explorer"

# 5. Test installation (optional)
python test_installation.py
```

## Usage

1. **Activate environment:** `conda activate encodec_explore`
2. **Start Jupyter:** `jupyter notebook` (or `jupyter lab`)
3. **Select kernel:** In your notebook: Kernel → Change Kernel → "Encodec Explorer"

## Alternative: Conda Environment File

If you prefer conda for package management:

```bash
conda env create -f environment.yml
conda activate encodec_explore
python -m ipykernel install --user --name encodec_explore --display-name "Encodec Explorer"
```

## Troubleshooting

**Kernel not showing in Jupyter:**
- Close Jupyter completely and restart
- Check: `jupyter kernelspec list` (should show `encodec_explore`)

**Packages missing:**
- Make sure you activated: `conda activate encodec_explore`
- Reinstall: `pip install -r requirements.txt`

**GPU support (optional):**
```bash
conda activate encodec_explore
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

In your notebook:
```python
import torch, soundfile, transformers
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("✅ Ready!")
```
