#!/usr/bin/env python3
"""
Live test of red team pipeline with actual Ollama model.
Tests with 5 prompts to verify the complete pipeline works.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.red_team.core.prompt_dataset import PromptDataset
from src.red_team.core.red_team_orchestrator import RedTeamOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test red team pipeline with 5 prompts."""
    
    print("="*70)
    print("RED TEAM LIVE TEST - 5 PROMPTS")
    print("="*70)
    
    # Load prompts
    print("\n1. Loading prompts...")
    dataset = PromptDataset()
    all_prompts = dataset.load_dataset("data/prompts.json")
    
    # Select 5 prompts from different categories
    test_prompts = [
        all_prompts[0],   # Cybercrime
        all_prompts[10],  # DAN
        all_prompts[20],  # Prompt Injection
        all_prompts[30],  # Violence
        all_prompts[40],  # Hate Speech
    ]
    
    print(f"✓ Selected {len(test_prompts)} test prompts:")
    for p in test_prompts:
        print(f"   - {p.id}: {p.category}")
    
    # Create orchestrator
    print("\n2. Initializing orchestrator...")
    orchestrator = RedTeamOrchestrator()
    print("✓ Orchestrator ready")
    
    # Test English tier only
    print("\n3. Testing English tier (this will take ~10-15 seconds)...")
    print("   Generating responses from llama3:8b...")
    
    try:
        results = orchestrator.test_linguistic_tier("english", test_prompts)
        
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Language: {results.language.upper()}")
        print(f"Total prompts: {results.total_prompts}")
        print(f"Unsafe responses: {results.unsafe_count}")
        print(f"Safe responses: {results.safe_count}")
        print(f"Error responses: {results.error_count}")
        print(f"Attack Success Rate: {results.asr:.2f}%")
        print("="*70)
        
        # Show sample responses
        print("\nSample Classifications:")
        for i, (resp, classif) in enumerate(zip(results.responses[:3], results.classifications[:3])):
            print(f"\n{i+1}. Prompt: {resp['prompt_text'][:60]}...")
            print(f"   Response: {resp['response_text'][:80]}...")
            print(f"   Classification: {'UNSAFE' if not classif['is_safe'] else 'SAFE'}")
            print(f"   Confidence: {classif['confidence']:.2f}")
        
        print("\n✓ Live test successful!")
        print("\nThe pipeline is working correctly with the actual model.")
        print("\nTo run the full experiment (100 prompts × 3 languages):")
        print("  python3 run_red_team.py")
        print("\nExpected time: ~10-15 minutes")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        logger.exception("Test failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
