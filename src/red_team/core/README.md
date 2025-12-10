# Red Team Core Components

This directory contains the core implementation of the red team testing pipeline.

## Components

### `prompt_dataset.py`
- **Purpose**: Dataset management and prompt handling
- **Classes**: `Prompt`, `PromptDataset`
- **Features**: Loading, validation, categorization of attack prompts

### `translation_engine.py`
- **Purpose**: Multilingual translation (English → Hindi/Hinglish)
- **Classes**: `TranslationEngine`
- **Features**: AI4Bharat IndicTrans2 integration, batch translation

### `ollama_interface.py`
- **Purpose**: Interface with Ollama-hosted LLM models
- **Classes**: `OllamaInterface`, `Response`
- **Features**: Model communication, response generation, error handling

### `safety_classifier.py`
- **Purpose**: Response safety classification
- **Classes**: `SafetyClassifier`, `Classification`
- **Features**: Rule-based safety evaluation, confidence scoring

### `red_team_orchestrator.py`
- **Purpose**: Pipeline coordination and experiment management
- **Classes**: `RedTeamOrchestrator`, `RedTeamResults`, `TierResults`
- **Features**: Multi-language testing, ASR calculation, results management

## Usage

```python
# Import from red team module (recommended)
from src.red_team import PromptDataset, RedTeamOrchestrator

# Or import directly from core
from src.red_team.core.prompt_dataset import PromptDataset
from src.red_team.core.red_team_orchestrator import RedTeamOrchestrator
```

## Architecture

The core components follow a modular design:

1. **Data Layer**: `PromptDataset` manages attack prompts
2. **Translation Layer**: `TranslationEngine` handles multilingual conversion
3. **Model Layer**: `OllamaInterface` communicates with LLM
4. **Classification Layer**: `SafetyClassifier` evaluates responses
5. **Orchestration Layer**: `RedTeamOrchestrator` coordinates the pipeline

Each component is designed to be independently testable and replaceable.