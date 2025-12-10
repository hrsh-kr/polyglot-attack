"""
Integration test for Red Team Orchestrator.

Tests that the orchestrator properly integrates with all components.
"""

from src.red_team.core.red_team_orchestrator import RedTeamOrchestrator
from src.red_team.core.prompt_dataset import Prompt
from src.red_team.core.ollama_interface import Response
from src.red_team.core.safety_classifier import Classification
from datetime import datetime


def test_orchestrator_integration():
    """Test that orchestrator integrates with all components."""
    print("Testing Red Team Orchestrator Integration...")
    
    # Create orchestrator
    orchestrator = RedTeamOrchestrator()
    print("✓ Orchestrator created successfully")
    
    # Test ASR calculation
    test_classifications = [
        Classification(
            response_id="test_1",
            is_safe=False,
            confidence=0.9,
            rationale="Test unsafe",
            requires_manual_review=False
        ),
        Classification(
            response_id="test_2",
            is_safe=True,
            confidence=0.9,
            rationale="Test safe",
            requires_manual_review=False
        ),
    ]
    
    asr = orchestrator.calculate_asr(test_classifications)
    assert asr == 50.0, f"Expected ASR of 50%, got {asr}%"
    print(f"✓ ASR calculation works: {asr}%")
    
    # Test that components are initialized
    assert orchestrator.ollama is not None, "Ollama interface not initialized"
    assert orchestrator.classifier is not None, "Classifier not initialized"
    assert orchestrator.translator is not None, "Translator not initialized"
    assert orchestrator.dataset_manager is not None, "Dataset manager not initialized"
    print("✓ All components initialized")
    
    # Test model info retrieval
    try:
        model_info = orchestrator.ollama.get_model_info()
        print(f"✓ Model info retrieved: {model_info.get('model_name', 'unknown')}")
    except Exception as e:
        print(f"⚠ Model info retrieval failed (Ollama may not be running): {e}")
    
    print("\n✓ Integration test passed!")
    return True


if __name__ == "__main__":
    test_orchestrator_integration()

