"""
Red Team Orchestrator Module

Coordinates the entire red team testing pipeline:
- Translation of prompts
- Testing across linguistic tiers
- Response classification
- ASR calculation and metrics
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime
import logging

import config
from .prompt_dataset import PromptDataset, Prompt
from .translation_engine import TranslationEngine
from .ollama_interface import OllamaInterface, Response
from .safety_classifier import SafetyClassifier, Classification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx request logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class TierResults:
    """
    Results for a single linguistic tier.
    
    Attributes:
        language: Language tier (english, hindi, hinglish)
        total_prompts: Total number of prompts tested
        responses: List of Response objects
        classifications: List of Classification objects
        unsafe_count: Number of unsafe responses
        safe_count: Number of safe responses
        error_count: Number of error responses
        asr: Attack Success Rate (percentage)
        timestamp: When testing was completed
    """
    language: str
    total_prompts: int
    responses: List[Dict]
    classifications: List[Dict]
    unsafe_count: int
    safe_count: int
    error_count: int
    asr: float
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TierResults':
        """Create TierResults instance from dictionary."""
        return cls(**data)


@dataclass
class RedTeamResults:
    """
    Complete red team testing results across all linguistic tiers.
    
    Attributes:
        total_prompts: Total number of unique prompts (100)
        english_asr: Attack Success Rate for English
        hindi_asr: Attack Success Rate for Hindi
        hinglish_asr: Attack Success Rate for Hinglish
        tier_results: Dictionary mapping language to TierResults
        model_info: Information about the model tested
        timestamp: When testing was completed
        output_file: Path where results were saved (optional)
    """
    total_prompts: int
    english_asr: float
    hindi_asr: float
    hinglish_asr: float
    tier_results: Dict[str, Dict]
    model_info: Dict
    timestamp: str
    output_file: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RedTeamResults':
        """Create RedTeamResults instance from dictionary."""
        return cls(**data)


class RedTeamOrchestrator:
    """
    Orchestrates the complete red team testing pipeline.
    
    Coordinates:
    1. Loading/translating prompts
    2. Testing each linguistic tier
    3. Classifying responses
    4. Calculating metrics (ASR)
    5. Saving results
    """

    def __init__(
        self,
        ollama_interface: Optional[OllamaInterface] = None,
        safety_classifier: Optional[SafetyClassifier] = None,
        translation_engine: Optional[TranslationEngine] = None
    ):
        """
        Initialize red team orchestrator.
        
        Args:
            ollama_interface: OllamaInterface instance (creates default if None)
            safety_classifier: SafetyClassifier instance (creates default if None)
            translation_engine: TranslationEngine instance (creates default if None)
        """
        self.ollama = ollama_interface or OllamaInterface()
        self.classifier = safety_classifier or SafetyClassifier()
        self.translator = translation_engine or TranslationEngine()
        self.dataset_manager = PromptDataset()

    def run_red_team_phase(
        self,
        prompts_file: Optional[str] = None,
        output_file: Optional[str] = None,
        skip_translation: bool = False,
        use_timestamp: bool = True
    ) -> RedTeamResults:
        """
        Run the complete red team testing phase.
        
        Args:
            prompts_file: Path to prompts file (uses config default if None)
            output_file: Path to save results (uses config default if None)
            skip_translation: If True, assumes prompts are already translated
            use_timestamp: If True, add timestamp to output filename (default)
            
        Returns:
            RedTeamResults object with complete results
        """
        logger.info("="*70)
        logger.info("RED TEAM PHASE - MULTILINGUAL SAFETY TESTING")
        logger.info("="*70)
        
        # Determine which prompts file to load
        if skip_translation:
            # When skipping translation, load from translated file
            if prompts_file is None or prompts_file == str(config.PROMPTS_FILE):
                prompts_file = str(config.PROMPTS_TRANSLATED_FILE)
                logger.info("Skip translation requested - loading translated prompts")
            else:
                logger.info("Skip translation requested - using specified file")
        else:
            # Normal mode - use base prompts file
            prompts_file = prompts_file or str(config.PROMPTS_FILE)
        
        logger.info(f"Loading prompts from {prompts_file}")
        
        try:
            prompts = self.dataset_manager.load_dataset(prompts_file)
            logger.info(f"✓ Loaded {len(prompts)} prompts")
        except FileNotFoundError:
            logger.error(f"Prompts file not found: {prompts_file}")
            if skip_translation:
                logger.error("When using --skip-translation, make sure translated prompts exist.")
                logger.error("Run without --skip-translation first to generate translations.")
            raise
        
        # Check if translation is needed
        needs_translation = any(
            p.text_hindi is None or p.text_hinglish is None 
            for p in prompts
        )
        
        if needs_translation and not skip_translation:
            logger.info("Prompts need translation - starting translation phase")
            prompts = self.translator.batch_translate(prompts)
            
            # Save translated prompts
            translated_file = str(config.PROMPTS_TRANSLATED_FILE)
            self.dataset_manager.save_dataset(prompts, translated_file)
            logger.info(f"✓ Saved translated prompts to {translated_file}")
        elif skip_translation:
            if needs_translation:
                logger.warning("Some prompts are missing translations, but skip_translation=True")
                logger.warning("Only prompts with existing translations will be tested")
            else:
                logger.info("✓ Using pre-translated prompts")
        else:
            logger.info("✓ Prompts already translated")
        
        # Get model info
        model_info = self.ollama.get_model_info()
        logger.info(f"✓ Testing model: {model_info['model_name']}")
        
        # Test each linguistic tier
        tier_results = {}
        
        for language in config.RED_TEAM_CONFIG["linguistic_tiers"]:
            print(f"\n{'='*70}")
            print(f"TESTING {language.upper()} TIER")
            print(f"{'='*70}")
            
            results = self.test_linguistic_tier(language, prompts)
            tier_results[language] = results.to_dict()
            
            print(f"\n✓ {language.upper()} tier complete:")
            print(f"  - Total prompts: {results.total_prompts}")
            print(f"  - Unsafe responses: {results.unsafe_count}")
            print(f"  - Safe responses: {results.safe_count}")
            print(f"  - Error responses: {results.error_count}")
            print(f"  - Attack Success Rate: {results.asr:.2f}%")
        
        # Create final results
        final_results = RedTeamResults(
            total_prompts=len(prompts),
            english_asr=tier_results["english"]["asr"],
            hindi_asr=tier_results["hindi"]["asr"],
            hinglish_asr=tier_results["hinglish"]["asr"],
            tier_results=tier_results,
            model_info=model_info,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        output_file = output_file or str(config.RED_TEAM_RESULTS_FILE)
        actual_output_file = self._save_results(final_results, output_file, use_timestamp)
        logger.info(f"\n✓ Saved red team results to {actual_output_file}")
        
        # Store actual output file path in results for reference
        final_results.output_file = actual_output_file
        
        # Print summary
        self._print_summary(final_results)
        
        return final_results

    def test_linguistic_tier(self, language: str, prompts: List[Prompt]) -> TierResults:
        """
        Test a single linguistic tier (English, Hindi, or Hinglish).
        
        Args:
            language: Language tier to test ("english", "hindi", "hinglish")
            prompts: List of Prompt objects with translations
            
        Returns:
            TierResults object with results for this tier
        """
        # Extract prompts for this language
        prompt_tuples = []
        for prompt in prompts:
            if language == "english":
                text = prompt.text_english
            elif language == "hindi":
                text = prompt.text_hindi
            elif language == "hinglish":
                text = prompt.text_hinglish
            else:
                raise ValueError(f"Unknown language tier: {language}")
            
            # Skip if translation is missing
            if text is None:
                logger.warning(f"Skipping {prompt.id} - no {language} translation")
                continue
            
            prompt_tuples.append((text, f"{prompt.id}_{language}"))
        
        print(f"Testing {len(prompt_tuples)} prompts...")
        
        # Generate responses with progress indicators
        print("Generating responses from model...")
        responses = []
        total = len(prompt_tuples)
        
        for i, (prompt_text, prompt_id) in enumerate(prompt_tuples, 1):
            progress_pct = (i / total) * 100
            print(f"[{progress_pct:5.1f}%] Prompt {i:3d}/{total} (ID: {prompt_id})...", end=' ', flush=True)
            
            try:
                response = self.ollama.generate_response(prompt_text, prompt_id)
                responses.append(response)
                
                # Simple status indicator
                if response.response_text.startswith("ERROR:"):
                    print("✗")
                else:
                    print("✓")
                    
            except Exception as e:
                print("✗")
                # Create error response
                error_response = Response(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    response_text=f"ERROR: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    model_name=self.ollama.model_name,
                    inference_params=self.ollama.inference_params.copy()
                )
                responses.append(error_response)
        
        print(f"\n✓ Generated {len(responses)} responses")
        
        # Classify responses
        print("Classifying responses...")
        classifications = self.classifier.batch_classify(responses)
        print(f"✓ Classified {len(classifications)} responses")
        
        # Calculate metrics
        unsafe_count = sum(1 for c in classifications if not c.is_safe)
        safe_count = sum(1 for c in classifications if c.is_safe)
        error_count = sum(1 for r in responses if r.response_text.startswith("ERROR:"))
        
        # Calculate ASR
        asr = self.calculate_asr(classifications)
        
        # Create tier results
        return TierResults(
            language=language,
            total_prompts=len(prompt_tuples),
            responses=[r.to_dict() for r in responses],
            classifications=[c.to_dict() for c in classifications],
            unsafe_count=unsafe_count,
            safe_count=safe_count,
            error_count=error_count,
            asr=asr,
            timestamp=datetime.now().isoformat()
        )

    def calculate_asr(self, classifications: List[Classification]) -> float:
        """
        Calculate Attack Success Rate from classifications.
        
        ASR = (Number of unsafe responses) / (Total responses) * 100
        
        Args:
            classifications: List of Classification objects
            
        Returns:
            Attack Success Rate as a percentage (0-100)
        """
        if not classifications:
            return 0.0
        
        unsafe_count = sum(1 for c in classifications if not c.is_safe)
        total_count = len(classifications)
        
        asr = (unsafe_count / total_count) * 100.0
        return asr

    def _save_results(self, results: RedTeamResults, filepath: str, use_timestamp: bool = True) -> str:
        """
        Save red team results to JSON file with optional timestamping.
        
        Args:
            results: RedTeamResults object
            filepath: Base path for the results file
            use_timestamp: If True, add timestamp to filename (default behavior)
            
        Returns:
            Actual filepath where results were saved
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename by default
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            timestamped_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
            actual_path = timestamped_path
        else:
            actual_path = path
        
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(actual_path)

    def _print_summary(self, results: RedTeamResults) -> None:
        """
        Print a summary of red team results.
        
        Args:
            results: RedTeamResults object
        """
        print("\n" + "="*70)
        print("RED TEAM PHASE SUMMARY")
        print("="*70)
        print(f"Model: {results.model_info['model_name']}")
        print(f"Total Prompts: {results.total_prompts}")
        print(f"Timestamp: {results.timestamp}")
        print("\nAttack Success Rates:")
        print(f"  English:  {results.english_asr:6.2f}%")
        print(f"  Hindi:    {results.hindi_asr:6.2f}%")
        print(f"  Hinglish: {results.hinglish_asr:6.2f}%")
        print("\nKey Finding:")
        
        # Determine vulnerability
        max_asr = max(results.english_asr, results.hindi_asr, results.hinglish_asr)
        if max_asr == results.hinglish_asr and results.hinglish_asr > results.english_asr:
            increase = results.hinglish_asr - results.english_asr
            print(f"  ⚠ Hinglish shows {increase:.1f}% higher ASR than English")
            print(f"  ⚠ Model is vulnerable to code-mixed attacks")
        elif max_asr == results.hindi_asr and results.hindi_asr > results.english_asr:
            increase = results.hindi_asr - results.english_asr
            print(f"  ⚠ Hindi shows {increase:.1f}% higher ASR than English")
            print(f"  ⚠ Model is vulnerable to Hindi attacks")
        else:
            print(f"  ✓ English baseline is highest - model maintains safety")
        
        print("="*70 + "\n")


def run_red_team_experiment(
    prompts_file: Optional[str] = None,
    output_file: Optional[str] = None,
    skip_translation: bool = False,
    use_timestamp: bool = True
) -> RedTeamResults:
    """
    Convenience function to run the complete red team experiment.
    
    Args:
        prompts_file: Path to prompts file (uses config default if None)
        output_file: Path to save results (uses config default if None)
        skip_translation: If True, assumes prompts are already translated
        use_timestamp: If True, add timestamp to output filename (default)
        
    Returns:
        RedTeamResults object
    """
    orchestrator = RedTeamOrchestrator()
    return orchestrator.run_red_team_phase(
        prompts_file=prompts_file,
        output_file=output_file,
        skip_translation=skip_translation,
        use_timestamp=use_timestamp
    )


if __name__ == "__main__":
    # Run the red team experiment
    results = run_red_team_experiment()
    print("\n✓ Red team experiment complete!")
    print(f"✓ Results saved to {config.RED_TEAM_RESULTS_FILE}")

