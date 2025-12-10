#!/usr/bin/env python3
"""
Red Team Pipeline Validation Script

Validates the red team pipeline components and checks data integrity.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.red_team.core.prompt_dataset import PromptDataset
from src.red_team.core.ollama_interface import OllamaInterface
from src.red_team.core.safety_classifier import SafetyClassifier
from src.red_team.core.red_team_orchestrator import RedTeamOrchestrator


def validate_prompts():
    """Validate prompt dataset."""
    print("\n" + "="*70)
    print("VALIDATING PROMPT DATASET")
    print("="*70)
    
    try:
        dataset = PromptDataset()
        prompts = dataset.load_dataset(str(config.PROMPTS_FILE))
        
        print(f"✓ Loaded {len(prompts)} prompts")
        
        # Validate structure
        if dataset.validate_dataset():
            print("✓ Dataset structure valid")
        else:
            print("✗ Dataset structure invalid")
            return False
        
        # Check categories
        categories = config.DATASET_CONFIG["categories"]
        for category in categories:
            count = len(dataset.get_prompts_by_category(category))
            expected = config.DATASET_CONFIG["prompts_per_category"]
            if count == expected:
                print(f"✓ {category}: {count} prompts")
            else:
                print(f"✗ {category}: {count} prompts (expected {expected})")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating prompts: {e}")
        return False


def validate_translations():
    """Validate translated prompts."""
    print("\n" + "="*70)
    print("VALIDATING TRANSLATIONS")
    print("="*70)
    
    try:
        if not Path(config.PROMPTS_TRANSLATED_FILE).exists():
            print("⚠ Translated prompts file not found")
            print("  Run translation to generate: python3 -c \"from src.red_team.core.translation_engine import translate_dataset; translate_dataset()\"")
            return False
        
        dataset = PromptDataset()
        prompts = dataset.load_dataset(str(config.PROMPTS_TRANSLATED_FILE))
        
        print(f"✓ Loaded {len(prompts)} translated prompts")
        
        # Check translation completeness
        hindi_count = sum(1 for p in prompts if p.text_hindi)
        hinglish_count = sum(1 for p in prompts if p.text_hinglish)
        
        print(f"✓ Hindi translations: {hindi_count}/{len(prompts)}")
        print(f"✓ Hinglish translations: {hinglish_count}/{len(prompts)}")
        
        if hindi_count == len(prompts) and hinglish_count == len(prompts):
            print("✓ All prompts translated")
            return True
        else:
            print("⚠ Some translations missing")
            return False
        
    except Exception as e:
        print(f"✗ Error validating translations: {e}")
        return False


def validate_ollama():
    """Validate Ollama connection."""
    print("\n" + "="*70)
    print("VALIDATING OLLAMA CONNECTION")
    print("="*70)
    
    try:
        ollama = OllamaInterface()
        model_info = ollama.get_model_info()
        
        if "error" in model_info:
            print(f"✗ Ollama error: {model_info['error']}")
            return False
        
        print(f"✓ Connected to Ollama")
        print(f"✓ Model: {model_info['model_name']}")
        print(f"✓ Temperature: {ollama.temperature}")
        print(f"✓ Max tokens: {ollama.max_tokens}")
        print(f"✓ Top-p: {ollama.top_p}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def validate_classifier():
    """Validate safety classifier."""
    print("\n" + "="*70)
    print("VALIDATING SAFETY CLASSIFIER")
    print("="*70)
    
    try:
        classifier = SafetyClassifier()
        
        print(f"✓ Classifier initialized")
        print(f"✓ Confidence threshold: {classifier.confidence_threshold}")
        print(f"✓ Unsafe indicators: {len(classifier.unsafe_indicators)}")
        print(f"✓ Safe indicators: {len(classifier.safe_indicators)}")
        
        # Test with sample responses
        from src.red_team.core.ollama_interface import Response
        from datetime import datetime
        
        test_safe = Response(
            prompt_id="test_safe",
            prompt_text="Test",
            response_text="I cannot assist with that request.",
            timestamp=datetime.now().isoformat(),
            model_name="test",
            inference_params={}
        )
        
        test_unsafe = Response(
            prompt_id="test_unsafe",
            prompt_text="Test",
            response_text="Here's how you can do it: First, you need to...",
            timestamp=datetime.now().isoformat(),
            model_name="test",
            inference_params={}
        )
        
        safe_classification = classifier.classify_response(test_safe)
        unsafe_classification = classifier.classify_response(test_unsafe)
        
        if safe_classification.is_safe and not unsafe_classification.is_safe:
            print("✓ Classifier correctly identifies safe/unsafe responses")
            return True
        else:
            print("✗ Classifier misclassified test responses")
            return False
        
    except Exception as e:
        print(f"✗ Error validating classifier: {e}")
        return False


def validate_orchestrator():
    """Validate red team orchestrator."""
    print("\n" + "="*70)
    print("VALIDATING RED TEAM ORCHESTRATOR")
    print("="*70)
    
    try:
        orchestrator = RedTeamOrchestrator()
        
        print(f"✓ Orchestrator initialized")
        print(f"✓ Ollama interface: {type(orchestrator.ollama).__name__}")
        print(f"✓ Safety classifier: {type(orchestrator.classifier).__name__}")
        print(f"✓ Translation engine: {type(orchestrator.translator).__name__}")
        print(f"✓ Dataset manager: {type(orchestrator.dataset_manager).__name__}")
        
        # Test ASR calculation
        from src.red_team.core.safety_classifier import Classification
        
        test_classifications = [
            Classification("1", False, 0.9, "unsafe", False),
            Classification("2", True, 0.9, "safe", False),
            Classification("3", True, 0.9, "safe", False),
        ]
        
        asr = orchestrator.calculate_asr(test_classifications)
        expected_asr = (1/3) * 100
        
        if abs(asr - expected_asr) < 0.01:
            print(f"✓ ASR calculation correct: {asr:.2f}%")
            return True
        else:
            print(f"✗ ASR calculation incorrect: {asr:.2f}% (expected {expected_asr:.2f}%)")
            return False
        
    except Exception as e:
        print(f"✗ Error validating orchestrator: {e}")
        return False


def validate_results():
    """Validate existing red team results."""
    print("\n" + "="*70)
    print("VALIDATING EXISTING RESULTS")
    print("="*70)
    
    try:
        if not Path(config.RED_TEAM_RESULTS_FILE).exists():
            print("⚠ No existing results found")
            print("  Run red team experiment to generate results")
            return None
        
        with open(config.RED_TEAM_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        
        print(f"✓ Loaded results from {config.RED_TEAM_RESULTS_FILE}")
        print(f"  Model: {results['model_info']['model_name']}")
        print(f"  Total prompts: {results['total_prompts']}")
        print(f"  Timestamp: {results['timestamp']}")
        
        print("\nAttack Success Rates:")
        print(f"  English:  {results['english_asr']:6.2f}%")
        print(f"  Hindi:    {results['hindi_asr']:6.2f}%")
        print(f"  Hinglish: {results['hinglish_asr']:6.2f}%")
        
        # Check data integrity
        all_valid = True
        for lang in ['english', 'hindi', 'hinglish']:
            tier = results['tier_results'][lang]
            
            if tier['total_prompts'] == 0:
                print(f"⚠ {lang}: No data (tier not run)")
                continue
            
            # Check counts
            total_classified = tier['unsafe_count'] + tier['safe_count']
            if total_classified != len(tier['classifications']):
                print(f"✗ {lang}: Classification count mismatch")
                all_valid = False
            
            # Check ASR
            expected_asr = (tier['unsafe_count'] / tier['total_prompts']) * 100
            if abs(tier['asr'] - expected_asr) > 0.01:
                print(f"✗ {lang}: ASR calculation incorrect")
                all_valid = False
        
        if all_valid:
            print("\n✓ Results data integrity validated")
            return True
        else:
            print("\n✗ Results data integrity issues found")
            return False
        
    except Exception as e:
        print(f"✗ Error validating results: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("RED TEAM PIPELINE VALIDATION")
    print("="*70)
    
    checks = [
        ("Prompt Dataset", validate_prompts),
        ("Translations", validate_translations),
        ("Ollama Connection", validate_ollama),
        ("Safety Classifier", validate_classifier),
        ("Red Team Orchestrator", validate_orchestrator),
        ("Existing Results", validate_results),
    ]
    
    results = {}
    for name, check_func in checks:
        result = check_func()
        results[name] = result
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{status:8s} {name}")
    
    # Overall status
    failed = sum(1 for r in results.values() if r is False)
    passed = sum(1 for r in results.values() if r is True)
    
    print("\n" + "="*70)
    if failed == 0:
        print("✓ ALL CRITICAL CHECKS PASSED")
        print("="*70)
        print("\nRed team pipeline is ready!")
        return 0
    else:
        print(f"✗ {failed} CHECK(S) FAILED, {passed} PASSED")
        print("="*70)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
