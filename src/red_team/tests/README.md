# Red Team Test Suite

This directory contains comprehensive tests for all red team components.

## Test Files

### `test_prompt_dataset.py`
- Tests prompt loading, validation, and categorization
- Verifies dataset structure and integrity
- Tests prompt filtering and selection

### `test_translation_engine.py`
- Tests Hindi and Hinglish translation functionality
- Verifies batch translation operations
- Tests translation quality and consistency

### `test_ollama_interface.py`
- Tests Ollama model communication
- Verifies response generation and error handling
- Tests model configuration and parameters

### `test_safety_classifier.py`
- Tests response classification accuracy
- Verifies confidence scoring
- Tests safe/unsafe detection patterns

### `test_red_team_orchestrator.py`
- Tests complete pipeline orchestration
- Verifies ASR calculation accuracy
- Tests multi-language experiment coordination

### `test_integration_orchestrator.py`
- End-to-end integration tests
- Tests complete workflow with real components
- Verifies data flow between components

## Running Tests

### All Tests
```bash
python3 -m pytest src/red_team/tests/ -v
```

### Individual Test Files
```bash
python3 -m pytest src/red_team/tests/test_prompt_dataset.py -v
python3 -m pytest src/red_team/tests/test_safety_classifier.py -v
```

### Quick Test
```bash
python3 scripts/test_red_team_live.py
```

## Test Coverage

The test suite provides comprehensive coverage:
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction
- **Live Tests**: Real model validation
- **Edge Cases**: Error conditions and boundary values

## Test Status

✅ **All tests passing**
- All core components tested
- Integration tests verified  
- Live model tests successful
- Comprehensive test coverage across all modules