"""
Red Team Module

Contains components for adversarial testing of LLM safety:
- Prompt dataset management
- Translation engine
- Ollama interface for model testing
- Safety classification
- Red team orchestration
"""

# Import core components for easy access
from .core import (
    PromptDataset,
    Prompt,
    TranslationEngine,
    OllamaInterface,
    Response,
    SafetyClassifier,
    Classification,
    RedTeamOrchestrator,
    RedTeamResults,
    TierResults,
)

__all__ = [
    'PromptDataset',
    'Prompt',
    'TranslationEngine',
    'OllamaInterface',
    'Response',
    'SafetyClassifier',
    'Classification',
    'RedTeamOrchestrator',
    'RedTeamResults',
    'TierResults',
]

