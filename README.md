# Multilingual LLM Safety Testing System

A Red-Blue Teaming framework to validate and mitigate multilingual safety vulnerabilities in locally-hosted Llama models running via Ollama.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## 🎯 Overview

This system implements a comprehensive Red-Blue Teaming framework to test multilingual safety vulnerabilities in LLMs:

**Red Team Phase**: Tests 100 high-risk prompts across 10 attack categories in 3 languages (English, Hindi, Hinglish) to quantify Attack Success Rate (ASR).

**Blue Team Phase**: Implements "The Polyglot Sentinel" - a middleware defense layer that normalizes multilingual inputs and applies safety filtering.

**Research Hypothesis**: Models aligned for English safety will show higher ASR when tested with Hinglish (code-mixed) prompts.

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: https://ollama.com/download
2. **Pull Llama model**:
   ```bash
   ollama pull llama3:8b
   ```
3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/version
   ```
4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

### Run Full Experiment (300 prompts)

```bash
# Run the red team experiment
python3 scripts/run_red_team.py --skip-translation
```

**Time**: ~10-15 minutes for 300 prompts (100 × 3 languages)

### Quick Test (5 prompts)

```bash
python3 scripts/test_red_team_live.py
```

## 🔧 Setup

For detailed setup instructions, see [Setup Guide](docs/SETUP_GUIDE.md).

### Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token from: https://huggingface.co/settings/tokens
   ```

3. **Verify system**:
   ```bash
   python3 scripts/validate_red_team.py
   ```

**Note**: Dataset and translations are already included in the `data/` directory.

**Important**: You'll need a Hugging Face token for translation features. Get one free at https://huggingface.co/settings/tokens

### Configuration

Edit `config.py` to customize:
- **Ollama settings**: Model name, temperature, max_tokens, top_p
- **File paths**: Data, results, and report locations
- **Classification**: Safety thresholds and indicators
- **Translation**: Language settings and service selection

## 📖 Usage

### Running Red Team Experiment

**Full experiment** (100 prompts × 3 languages = 300 tests):
```bash
python3 scripts/run_red_team.py --skip-translation
```

**Options**:
```bash
# Use custom prompts file
python3 scripts/run_red_team.py --prompts data/my_prompts.json

# Save results to custom location
python3 scripts/run_red_team.py --output results/my_results.json

# Skip translation (if already done)
python3 scripts/run_red_team.py --skip-translation
```

### Viewing Results

**Quick summary** (use latest timestamped file):
```bash
# Find latest results file
ls -t results/red_team_results_*.json | head -1

# View summary (replace with actual filename)
python3 -c "import json; d=json.load(open('results/red_team_results_YYYYMMDD_HHMM.json')); \
print(f'English ASR: {d[\"english_asr\"]:.2f}%'); \
print(f'Hindi ASR: {d[\"hindi_asr\"]:.2f}%'); \
print(f'Hinglish ASR: {d[\"hinglish_asr\"]:.2f}%')"
```

**Full results**:
```bash
# View latest results (replace with actual filename)
cat results/red_team_results_YYYYMMDD_HHMM.json | python3 -m json.tool | less
```

## 📁 Project Structure

```
multilingual-llm-safety-testing/
│
├── 📁 scripts/                    # 🚀 Executable Scripts
│   ├── run_red_team.py              # Main experiment runner
│   ├── test_red_team_live.py        # Quick 5-prompt test
│   ├── validate_red_team.py         # System validation
│   └── README.md                   # Scripts documentation
│
├── 📁 docs/                       # 📚 Documentation
│   ├── RED_TEAM_VALIDATION_REPORT.md
│   └── Research_Proposal_hk.pdf    # Research proposal
│
├── 📁 src/                        # 💻 Source Code
│   ├── red_team/                   # Red team implementation
│   │   ├── core/                     # Core implementation
│   │   │   ├── prompt_dataset.py       # Dataset management
│   │   │   ├── translation_engine.py   # Translation (Hindi/Hinglish)
│   │   │   ├── ollama_interface.py     # Model interface
│   │   │   ├── safety_classifier.py    # Response classification
│   │   │   └── red_team_orchestrator.py # Pipeline coordination
│   │   ├── tests/                    # Test suite
│   │   │   └── test_*.py               # Unit tests (32 tests)
│   │   └── __init__.py              # Module exports
│   │
│   └── blue_team/                  # Blue team (future)
│
├── 📁 data/                       # 📊 Data Files
│   ├── prompts.json                # 100 base prompts
│   └── prompts_translated.json     # Hindi + Hinglish translations
│
├── 📁 results/                    # 📈 Timestamped Results
├── 📁 reports/                    # 📋 Generated Reports
│   └── visualizations/             # Charts and graphs
├── 📁 .kiro/                      # ⚙️ Kiro Specifications
│
├── config.py                      # ⚙️ System configuration
├── requirements.txt               # 📦 Python dependencies
├── .env.example                   # 🔐 Environment variables template
├── README.md                      # 📖 This file
└── .gitignore                     # 🚫 Git ignore rules
```

### Key Components

**Red Team Pipeline**:
1. **Prompt Dataset** (100 prompts across 10 categories)
2. **Translation Engine** (Hindi + Hinglish generation)
3. **Ollama Interface** (Model communication)
4. **Safety Classifier** (Response evaluation)
5. **Orchestrator** (Pipeline coordination)

**Attack Categories**:
- Cybercrime
- DAN (Do Anything Now)
- Prompt Injection
- Violence
- Hate Speech
- Illegal Activities
- Misinformation
- Privacy Violation
- Manipulation
- Harmful Instructions

## 🧪 Testing

### Run All Tests

```bash
python3 -m pytest src/red_team/ -v
```

**Status**: ✅ 32/32 tests passing

### Test Components

```bash
# Test prompt dataset
python3 -m pytest src/red_team/tests/test_prompt_dataset.py -v

# Test safety classifier
python3 -m pytest src/red_team/tests/test_safety_classifier.py -v

# Test orchestrator
python3 -m pytest src/red_team/tests/test_red_team_orchestrator.py -v

# Test Ollama interface
python3 -m pytest src/red_team/tests/test_ollama_interface.py -v

# Integration test
python3 -m pytest src/red_team/tests/test_integration_orchestrator.py -v
```

### System Validation

```bash
# Full system validation
python3 scripts/validate_red_team.py

# Quick pipeline test (5 prompts)
python3 scripts/test_red_team_live.py
```

The validation script checks all components, while the live test verifies the complete pipeline works.

## 📊 Results

### Current Status

✅ **All components tested and working**
- 32/32 unit tests passing
- Live model test successful
- Translations complete (100 Hindi + 100 Hinglish)
- JSON serialization fixed
- Ready for full experiment

### Live Test Results (5 prompts)

- **Model**: llama3:8b
- **English ASR**: 0.00%
- **Status**: Strong safety baseline

Sample responses show strong safety:
- Cybercrime: "I cannot provide information or guidance on illegal activities..."
- DAN Jailbreak: "I cannot provide instructions on how to make explosives..."
- Prompt Injection: "I cannot provide instructions on how to make illegal substances..."

### Expected Full Results

Based on the strong English baseline (0% ASR), we expect:
- **English**: 0-10% ASR (strong safety filters)
- **Hindi**: 10-30% ASR (moderate vulnerability)
- **Hinglish**: 30-60% ASR (hypothesis: code-mixing bypasses filters)

### Understanding ASR

**Attack Success Rate (ASR)** = (Number of unsafe responses / Total responses) × 100%

- **Low ASR (0-20%)**: Model has strong safety filters
- **Medium ASR (20-50%)**: Model has moderate safety
- **High ASR (50-100%)**: Model is vulnerable to attacks

## 🔍 Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check available models
ollama list

# If not running, start it (usually automatic on macOS)
ollama serve
```

### Model Not Found

```bash
# Pull the model
ollama pull llama3:8b

# Verify it's available
ollama list | grep llama3
```

### Translation Failed

The translation engine uses AI4Bharat IndicTrans2 for high-quality translations. If translation fails:
1. **Check HF_TOKEN**: Ensure `.env` file exists with valid Hugging Face token
2. **Verify token access**: Check your token has access to gated models at https://huggingface.co/settings/tokens
3. **Check internet connection**: Model downloads require internet access
4. **Skip translation**: Use `--skip-translation` if translations already exist
5. **Use English-only**: Run the quick test for English-only testing

### Environment Variable Issues

```bash
# Check if .env file exists
ls -la .env

# Verify HF_TOKEN is loaded
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('HF_TOKEN:', 'SET' if os.getenv('HF_TOKEN') else 'NOT SET')"

# Create .env from template
cp .env.example .env
# Then edit .env and add your token
```

### Out of Memory

If testing 300 prompts causes memory issues:
1. Test in batches using custom prompt files
2. Reduce batch size in `config.py`
3. Use the quick test script instead

### Tests Failing

```bash
# Run tests with verbose output
python3 -m pytest src/red_team/ -v --tb=short

# Run specific test file
python3 -m pytest src/red_team/tests/test_prompt_dataset.py -v

# Quick validation
python3 scripts/test_red_team_live.py
```

## 🎯 Next Steps

After completing red team testing:

1. **Review Results**: Analyze latest timestamped results file (e.g., `results/red_team_results_YYYYMMDD_HHMM.json`)
2. **Compare ASR**: Examine differences between languages
3. **Identify Patterns**: Which attack categories are most successful?
4. **Validate Hypothesis**: Is Hinglish ASR > Hindi ASR > English ASR?
5. **Proceed to Blue Team**: Implement Polyglot Sentinel (Task 8)
6. **Run Validation**: Measure Safety Recovery Rate (SRR)
7. **Generate Report**: Create publication-ready analysis

## 📈 Performance Notes

- **Time per prompt**: ~2-3 seconds
- **5 prompts**: ~15 seconds
- **100 prompts**: ~3-5 minutes per language
- **300 prompts (full)**: ~10-15 minutes
- **Translation**: ~2-3 minutes (if needed)

## ⚠️ Safety Notes

**Warning**: This test suite contains harmful prompts designed to test model safety.

- Do not use these prompts for malicious purposes
- Results are for research purposes only
- Keep results confidential
- Follow ethical AI research guidelines

## 🔐 Security Notes

**Environment Variables**: This project uses environment variables to securely store API tokens.

- **Never commit `.env` files** to version control
- **Use `.env.example`** as a template for required variables
- **Keep your HF_TOKEN secure** - it provides access to your Hugging Face account
- **Regenerate tokens** if accidentally exposed

## 📝 Project Timeline

This project is designed for a 2-day implementation timeline:
- **Day 1**: Red team pipeline and initial testing ✅
- **Day 2**: Blue team implementation and report generation

## 🔬 Research Context

**Hypothesis**: Models aligned for safety in English exhibit catastrophic failures when processing Hinglish (Code-Mixed Hindi-English).

**Methodology**: 
1. Test 100 prompts in English (baseline)
2. Test same prompts in Hindi (translation effect)
3. Test same prompts in Hinglish (code-mixing effect)
4. Compare ASR across languages
5. Implement and validate defense (Polyglot Sentinel)

**Expected Contribution**: Quantitative evidence of multilingual vulnerability and a practical defense solution.

## 📄 License

Research project - see institution guidelines.

---

## 📚 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation and configuration
- **[Project Status](docs/PROJECT_STATUS.md)** - Current implementation status  
- **[Validation Report](docs/RED_TEAM_VALIDATION_REPORT.md)** - System validation results
- **Research Papers** - See `docs/` directory for additional documentation

---

**Status**: ✅ Ready for full experiment  
**Command**: `python3 scripts/run_red_team.py --skip-translation`  
**Expected Time**: 10-15 minutes  
**Expected Output**: Quantitative evidence of multilingual vulnerability
