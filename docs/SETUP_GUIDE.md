# Setup Guide

Complete setup guide for the Multilingual LLM Safety Testing System.

## Prerequisites

### 1. System Requirements
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space

### 2. Install Ollama
1. Download from: https://ollama.com/download
2. Install following platform-specific instructions
3. Verify installation:
   ```bash
   ollama --version
   ```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your Hugging Face token:
   ```bash
   # Get your token from: https://huggingface.co/settings/tokens
   HF_TOKEN=your_actual_token_here
   ```

## Setup Steps

### Step 1: Start Ollama Service
```bash
# Start Ollama (usually starts automatically)
ollama serve

# Verify it's running
curl http://localhost:11434/api/version
```

### Step 2: Pull Required Model
```bash
# Pull Llama 3 8B model (this may take several minutes)
ollama pull llama3:8b

# Verify model is available
ollama list | grep llama3
```

### Step 3: Configure Environment Variables
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your Hugging Face token:
   ```bash
   # Get your token from: https://huggingface.co/settings/tokens
   HF_TOKEN=your_actual_token_here
   ```

### Step 4: Verify Data Files
The dataset and translations are already included:
- `data/prompts.json` - 100 prompts across 10 attack categories
- `data/prompts_translated.json` - Hindi and Hinglish translations

### Step 5: Validate Setup
```bash
# Run full system validation
python3 scripts/validate_red_team.py
```

## Quick Test

Run a quick test to verify everything works:

```bash
# Test with 5 prompts (takes ~30 seconds)
python3 scripts/test_red_team_live.py
```

## Full Experiment

Once setup is complete, run the full experiment:

```bash
# Run complete red team experiment (300 prompts, ~10-15 minutes)
python3 scripts/run_red_team.py --skip-translation
```

## Troubleshooting

### Ollama Issues
- **Service not running**: Run `ollama serve`
- **Model not found**: Run `ollama pull llama3:8b`
- **Connection refused**: Check if port 11434 is available

### Python Issues
- **Import errors**: Ensure you're running from project root
- **Module not found**: Check Python path and virtual environment
- **Permission errors**: Use `python3` instead of `python`

### Environment Variable Issues
- **HF_TOKEN not found**: Ensure `.env` file exists with `HF_TOKEN=your_token`
- **Translation fails**: Check your Hugging Face token is valid at https://huggingface.co/settings/tokens
- **Permission denied**: Ensure your HF token has access to gated models

### Dataset Issues
- **File not found**: Dataset should exist at `data/prompts.json`
- **Translation missing**: Translations should exist at `data/prompts_translated.json`
- **Validation fails**: Check file permissions and paths

### Memory Issues
- **Out of memory**: Reduce batch size in `config.py`
- **Slow performance**: Close other applications
- **Process killed**: Use smaller model or increase swap space

## Configuration

Edit `config.py` to customize:

### Model Settings
```python
OLLAMA_CONFIG = {
    "model_name": "llama3:8b",  # Change model
    "temperature": 0.7,         # Adjust creativity
    "max_tokens": 512,          # Response length
    "top_p": 0.9,              # Nucleus sampling
}
```

### Dataset Settings
```python
DATASET_CONFIG = {
    "total_prompts": 100,       # Number of prompts
    "prompts_per_category": 10, # Prompts per attack type
}
```

### File Paths
```python
# Customize file locations
PROMPTS_FILE = DATA_DIR / "prompts.json"
RESULTS_DIR = PROJECT_ROOT / "results"
```

## Verification Checklist

- [ ] Ollama installed and running
- [ ] Model `llama3:8b` downloaded
- [ ] Python dependencies installed
- [ ] Environment variables configured (`.env` file with HF_TOKEN)
- [ ] Dataset created (`data/prompts.json`)
- [ ] Translations created (`data/prompts_translated.json`)
- [ ] Validation passes (`python3 scripts/validate_red_team.py`)
- [ ] Quick test works (`python3 scripts/test_red_team_live.py`)

## Next Steps

After successful setup:

1. **Run Experiment**: `python3 scripts/run_red_team.py --skip-translation`
2. **Analyze Results**: Check `results/` directory for timestamped output
3. **Review Documentation**: See `docs/` for detailed information
4. **Run Tests**: Use `python3 -m pytest src/red_team/tests/ -v`

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Run validation: `python3 scripts/validate_red_team.py`
3. Review logs and error messages
4. Ensure all prerequisites are met