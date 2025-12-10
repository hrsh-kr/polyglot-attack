#!/usr/bin/env python3
"""
Command-line script for running red team experiments.

Usage:
    python run_red_team.py [--prompts PROMPTS_FILE] [--output OUTPUT_FILE] [--skip-translation]
"""

import argparse
import sys
import logging
from pathlib import Path

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.red_team.core.red_team_orchestrator import run_red_team_experiment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for red team experiment."""
    parser = argparse.ArgumentParser(
        description="Run red team multilingual safety testing experiment"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(config.PROMPTS_FILE),
        help=f"Path to prompts JSON file (default: {config.PROMPTS_FILE})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.RED_TEAM_RESULTS_FILE),
        help=f"Path to save results JSON file (default: {config.RED_TEAM_RESULTS_FILE})"
    )
    
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation phase (assumes prompts are already translated)"
    )
    
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't add timestamp to output filename (may overwrite existing files)"
    )
    
    args = parser.parse_args()
    
    # Verify prompts file exists
    if not Path(args.prompts).exists():
        logger.error(f"Prompts file not found: {args.prompts}")
        logger.error("Please create the dataset first by running:")
        logger.error("  Dataset should already exist in data/prompts.json")
        sys.exit(1)
    
    try:
        # Run red team experiment
        logger.info("Starting red team experiment...")
        results = run_red_team_experiment(
            prompts_file=args.prompts,
            output_file=args.output,
            skip_translation=args.skip_translation,
            use_timestamp=not args.no_timestamp
        )
        
        logger.info("\n" + "="*70)
        logger.info("RED TEAM EXPERIMENT COMPLETE")
        logger.info("="*70)
        logger.info(f"Results saved to: {results.output_file or args.output}")
        logger.info(f"English ASR:  {results.english_asr:.2f}%")
        logger.info(f"Hindi ASR:    {results.hindi_asr:.2f}%")
        logger.info(f"Hinglish ASR: {results.hinglish_asr:.2f}%")
        logger.info("="*70)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

