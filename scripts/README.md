# Scripts Directory

This directory contains executable scripts for the multilingual LLM safety testing system.

## Scripts

### `run_red_team.py`
Main script for running red team experiments.

```bash
# Run full experiment with translation
python scripts/run_red_team.py

# Skip translation (faster, requires existing translations)
python scripts/run_red_team.py --skip-translation

# Custom output file
python scripts/run_red_team.py --output custom_results.json

# Disable timestamping (overwrite mode)
python scripts/run_red_team.py --no-timestamp
```

### `test_red_team_live.py`
Quick test script with 5 prompts to verify the pipeline works.

```bash
python scripts/test_red_team_live.py
```

### `validate_red_team.py`
Validation script to check all pipeline components.

```bash
python scripts/validate_red_team.py
```

## Usage Notes

- All scripts should be run from the project root directory
- Make sure Ollama is running before executing tests
- Scripts automatically handle import paths from the parent directory