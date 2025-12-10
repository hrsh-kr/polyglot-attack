"""
Unit tests for Red Team Orchestrator.

Tests core functionality of the orchestrator including:
- ASR calculation
- Tier testing coordination
- Results aggregation
"""

import pytest
from datetime import datetime
from src.red_team.core.red_team_orchestrator import RedTeamOrchestrator, TierResults, RedTeamResults
from src.red_team.core.safety_classifier import Classification
from src.red_team.core.ollama_interface import Response
from src.red_team.core.prompt_dataset import Prompt


class TestRedTeamOrchestrator:
    """Test suite for RedTeamOrchestrator class."""

    def test_calculate_asr_all_unsafe(self):
        """Test ASR calculation when all responses are unsafe."""
        orchestrator = RedTeamOrchestrator()
        
        classifications = [
            Classification(
                response_id=f"test_{i}",
                is_safe=False,
                confidence=0.9,
                rationale="Unsafe content detected",
                requires_manual_review=False
            )
            for i in range(10)
        ]
        
        asr = orchestrator.calculate_asr(classifications)
        assert asr == 100.0, "ASR should be 100% when all responses are unsafe"

    def test_calculate_asr_all_safe(self):
        """Test ASR calculation when all responses are safe."""
        orchestrator = RedTeamOrchestrator()
        
        classifications = [
            Classification(
                response_id=f"test_{i}",
                is_safe=True,
                confidence=0.9,
                rationale="Safe refusal detected",
                requires_manual_review=False
            )
            for i in range(10)
        ]
        
        asr = orchestrator.calculate_asr(classifications)
        assert asr == 0.0, "ASR should be 0% when all responses are safe"

    def test_calculate_asr_mixed(self):
        """Test ASR calculation with mixed safe/unsafe responses."""
        orchestrator = RedTeamOrchestrator()
        
        classifications = [
            Classification(
                response_id=f"test_{i}",
                is_safe=(i % 2 == 0),  # Alternate safe/unsafe
                confidence=0.9,
                rationale="Test classification",
                requires_manual_review=False
            )
            for i in range(10)
        ]
        
        asr = orchestrator.calculate_asr(classifications)
        assert asr == 50.0, "ASR should be 50% with half unsafe responses"

    def test_calculate_asr_empty_list(self):
        """Test ASR calculation with empty classification list."""
        orchestrator = RedTeamOrchestrator()
        
        asr = orchestrator.calculate_asr([])
        assert asr == 0.0, "ASR should be 0% for empty list"

    def test_calculate_asr_precision(self):
        """Test ASR calculation precision with various counts."""
        orchestrator = RedTeamOrchestrator()
        
        # 3 out of 7 unsafe = 42.857...%
        classifications = [
            Classification(
                response_id=f"test_{i}",
                is_safe=(i >= 3),
                confidence=0.9,
                rationale="Test",
                requires_manual_review=False
            )
            for i in range(7)
        ]
        
        asr = orchestrator.calculate_asr(classifications)
        expected = (3 / 7) * 100
        assert abs(asr - expected) < 0.01, f"ASR should be {expected:.2f}%"

    def test_tier_results_serialization(self):
        """Test TierResults can be serialized and deserialized."""
        tier_results = TierResults(
            language="english",
            total_prompts=10,
            responses=[],
            classifications=[],
            unsafe_count=3,
            safe_count=7,
            error_count=0,
            asr=30.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Serialize
        data = tier_results.to_dict()
        assert isinstance(data, dict)
        assert data["language"] == "english"
        assert data["asr"] == 30.0
        
        # Deserialize
        restored = TierResults.from_dict(data)
        assert restored.language == tier_results.language
        assert restored.asr == tier_results.asr

    def test_red_team_results_serialization(self):
        """Test RedTeamResults can be serialized and deserialized."""
        results = RedTeamResults(
            total_prompts=100,
            english_asr=20.0,
            hindi_asr=45.0,
            hinglish_asr=60.0,
            tier_results={},
            model_info={"model_name": "test_model"},
            timestamp=datetime.now().isoformat()
        )
        
        # Serialize
        data = results.to_dict()
        assert isinstance(data, dict)
        assert data["english_asr"] == 20.0
        assert data["hinglish_asr"] == 60.0
        
        # Deserialize
        restored = RedTeamResults.from_dict(data)
        assert restored.english_asr == results.english_asr
        assert restored.hinglish_asr == results.hinglish_asr

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized with default components."""
        orchestrator = RedTeamOrchestrator()
        
        assert orchestrator.ollama is not None
        assert orchestrator.classifier is not None
        assert orchestrator.translator is not None
        assert orchestrator.dataset_manager is not None

    def test_orchestrator_custom_components(self):
        """Test orchestrator can be initialized with custom components."""
        from src.red_team.core.ollama_interface import OllamaInterface
        from src.red_team.core.safety_classifier import SafetyClassifier
        from src.red_team.core.translation_engine import TranslationEngine
        
        custom_ollama = OllamaInterface()
        custom_classifier = SafetyClassifier()
        custom_translator = TranslationEngine()
        
        orchestrator = RedTeamOrchestrator(
            ollama_interface=custom_ollama,
            safety_classifier=custom_classifier,
            translation_engine=custom_translator
        )
        
        assert orchestrator.ollama is custom_ollama
        assert orchestrator.classifier is custom_classifier
        assert orchestrator.translator is custom_translator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

