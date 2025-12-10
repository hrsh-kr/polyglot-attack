"""
Unit tests for Prompt Dataset Management Module
"""

import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.red_team.core.prompt_dataset import Prompt, PromptDataset
import config


def test_prompt_creation():
    """Test creating a Prompt instance."""
    prompt = Prompt(
        id="test_01",
        category="Cybercrime",
        text_english="Test prompt",
        severity="high",
        expected_safe_response="Refusal"
    )
    assert prompt.id == "test_01"
    assert prompt.category == "Cybercrime"
    assert prompt.text_english == "Test prompt"
    assert prompt.text_hindi is None
    assert prompt.text_hinglish is None
    print("✓ test_prompt_creation passed")


def test_prompt_serialization():
    """Test Prompt to_dict and from_dict methods."""
    original = Prompt(
        id="test_02",
        category="Violence",
        text_english="Test prompt 2",
        text_hindi="टेस्ट",
        text_hinglish="Test टेस्ट",
        severity="critical",
        expected_safe_response="Refusal with explanation"
    )
    
    # Convert to dict and back
    prompt_dict = original.to_dict()
    restored = Prompt.from_dict(prompt_dict)
    
    assert restored.id == original.id
    assert restored.category == original.category
    assert restored.text_english == original.text_english
    assert restored.text_hindi == original.text_hindi
    assert restored.text_hinglish == original.text_hinglish
    assert restored.severity == original.severity
    print("✓ test_prompt_serialization passed")


def test_dataset_creation():
    """Test creating a full dataset."""
    dataset = PromptDataset()
    prompts = dataset.create_dataset()
    
    # Check total count
    assert len(prompts) == 100, f"Expected 100 prompts, got {len(prompts)}"
    
    # Check category distribution
    categories = config.DATASET_CONFIG["categories"]
    for category in categories:
        category_prompts = [p for p in prompts if p.category == category]
        assert len(category_prompts) == 10, f"Expected 10 prompts for {category}, got {len(category_prompts)}"
    
    # Check all prompts have required fields
    for prompt in prompts:
        assert prompt.id
        assert prompt.category
        assert prompt.text_english
        assert prompt.severity in ["high", "critical"]
        assert prompt.expected_safe_response
    
    print("✓ test_dataset_creation passed")


def test_dataset_validation():
    """Test dataset validation."""
    dataset = PromptDataset()
    dataset.create_dataset()
    
    # Valid dataset should pass
    assert dataset.validate_dataset() == True
    
    # Remove a prompt - should fail
    dataset.prompts.pop()
    assert dataset.validate_dataset() == False
    
    print("✓ test_dataset_validation passed")


def test_save_and_load():
    """Test saving and loading dataset."""
    # Create dataset
    dataset1 = PromptDataset()
    prompts1 = dataset1.create_dataset()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        dataset1.save_dataset(prompts1, temp_path)
        
        # Load from file
        dataset2 = PromptDataset()
        prompts2 = dataset2.load_dataset(temp_path)
        
        # Verify loaded data matches
        assert len(prompts2) == len(prompts1)
        assert prompts2[0].id == prompts1[0].id
        assert prompts2[0].text_english == prompts1[0].text_english
        assert prompts2[-1].category == prompts1[-1].category
        
        print("✓ test_save_and_load passed")
    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)


def test_get_prompts_by_category():
    """Test filtering prompts by category."""
    dataset = PromptDataset()
    dataset.create_dataset()
    
    # Test each category
    for category in config.DATASET_CONFIG["categories"]:
        category_prompts = dataset.get_prompts_by_category(category)
        assert len(category_prompts) == 10
        assert all(p.category == category for p in category_prompts)
    
    # Test non-existent category
    empty = dataset.get_prompts_by_category("NonExistent")
    assert len(empty) == 0
    
    print("✓ test_get_prompts_by_category passed")


def test_dataset_structure():
    """Test that dataset has correct structure per requirements."""
    dataset = PromptDataset()
    prompts = dataset.create_dataset()
    
    # Requirement 1.1: exactly 100 prompts
    assert len(prompts) == 100
    
    # Requirement 1.2: 10 prompts per category, all 10 categories present
    categories = config.DATASET_CONFIG["categories"]
    assert len(categories) == 10
    for category in categories:
        cat_prompts = [p for p in prompts if p.category == category]
        assert len(cat_prompts) == 10
    
    # Requirement 1.5: metadata present (category, severity, expected_safe_response)
    for prompt in prompts:
        assert prompt.category in categories
        assert prompt.severity in ["high", "critical"]
        assert prompt.expected_safe_response
    
    print("✓ test_dataset_structure passed")


if __name__ == "__main__":
    print("Running unit tests for Prompt Dataset...\n")
    
    test_prompt_creation()
    test_prompt_serialization()
    test_dataset_creation()
    test_dataset_validation()
    test_save_and_load()
    test_get_prompts_by_category()
    test_dataset_structure()
    
    print("\n✓ All tests passed!")
