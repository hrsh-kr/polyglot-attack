"""
Translation Engine Module

Handles translation of English prompts to Hindi and Hinglish for multilingual testing.
Uses AI4Bharat's IndicTrans2 - state-of-the-art model for Indian languages.
"""

from typing import List, Optional, Dict
import re
import time
import logging
import torch
import os
import config
from .prompt_dataset import Prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace token for gated models
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Please set it in your .env file or environment. "
        "Get your token from: https://huggingface.co/settings/tokens"
    )

# Lazy import for models (only load when needed)
_translation_pipeline = None


def get_indictrans2_pipeline(verbose: bool = True):
    """
    Load AI4Bharat IndicTrans2 pipeline with proper configuration.
    
    IndicTrans2 is specifically designed for Indian languages and provides
    superior translation quality compared to general-purpose models.
    
    Args:
        verbose: If True, show detailed loading messages
    """
    global _translation_pipeline
    
    if _translation_pipeline is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
        
        if verbose:
            logger.info("="*70)
            logger.info("Loading AI4Bharat IndicTrans2 (en→hi)")
            logger.info("State-of-the-art model for English→Hindi translation")
            logger.info("="*70)
        
        # Device configuration
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            logger.info(f"Using device: {DEVICE}")
        
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        src_lang, tgt_lang = "eng_Latn", "hin_Deva"
        
        # Load tokenizer
        if verbose:
            logger.info("Loading IndicTrans2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        # Load model with optimizations
        if verbose:
            logger.info("Loading IndicTrans2 model (1B parameters)...")
        if DEVICE == "cuda":
            # GPU with optimizations
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=HF_TOKEN
            ).to(DEVICE)
        else:
            # CPU
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=HF_TOKEN
            ).to(DEVICE)
        
        # Load IndicProcessor for preprocessing/postprocessing
        if verbose:
            logger.info("Loading IndicProcessor...")
        ip = IndicProcessor(inference=True)
        
        _translation_pipeline = {
            'tokenizer': tokenizer,
            'model': model,
            'processor': ip,
            'device': DEVICE,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang
        }
        
        if verbose:
            logger.info("✓ IndicTrans2 loaded successfully!")
            logger.info("="*70)
    
    return _translation_pipeline


# Enhanced Devanagari to Latin transliteration map
DEVANAGARI_TO_LATIN = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ॠ': 'ree', 'ऌ': 'lri', 'ए': 'e', 'ऐ': 'ai', 
    'ओ': 'o', 'औ': 'au',
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'w': 'w',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    # Vowel signs
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'ॄ': 'ree', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    # Special
    'ं': 'n', 'ः': 'h', '्': '', 'ँ': '', 'ॅ': 'e',
    'ऑ': 'o', 'ॉ': 'o', 'ड़': 'r', 'ढ़': 'rh',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f',
    '।': '.', '॥': '.',
    # Conjuncts
    'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gya', 'श्र': 'shr'
}


class TranslationEngine:
    """
    Advanced translation engine using AI4Bharat IndicTrans2.
    
    Features:
    - State-of-the-art translation quality for Indian languages
    - Proper handling of Indic scripts with IndicProcessor
    - Natural Hinglish generation
    - Technical term preservation
    - Batch processing with caching
    """

    def __init__(self):
        """Initialize translation engine."""
        self._translation_cache: Dict[str, str] = {}
        self._pipeline = None
        
        # Technical vocabulary to preserve in English
        self.technical_words = {
            'email', 'account', 'password', 'username', 'login',
            'website', 'url', 'browser', 'internet', 'online',
            'database', 'data', 'server', 'client', 'file',
            'hack', 'hacking', 'hacker', 'malware', 'virus',
            'ransomware', 'spyware', 'phishing', 'exploit',
            'sql', 'injection', 'attack', 'firewall', 'encryption',
            'computer', 'laptop', 'software', 'hardware', 'system',
            'app', 'application', 'program', 'code', 'network',
            'cyber', 'security', 'digital', 'cloud', 'api'
        }

    def initialize_model(self, verbose: bool = True) -> None:
        """
        Pre-initialize the IndicTrans2 model.
        
        This can be called before batch processing to load the model
        upfront and avoid initialization during the first translation.
        
        Args:
            verbose: If True, show detailed loading messages
        """
        if self._pipeline is None:
            if verbose:
                print("Initializing IndicTrans2 model...")
            self._get_pipeline(verbose=verbose)
            if verbose:
                print("✓ IndicTrans2 model loaded successfully!")
        elif verbose:
            print("✓ IndicTrans2 model already initialized")

    def is_initialized(self) -> bool:
        """Check if the translation model is already initialized."""
        return self._pipeline is not None

    def _get_pipeline(self, verbose: bool = False):
        """Get or initialize the translation pipeline."""
        if self._pipeline is None:
            self._pipeline = get_indictrans2_pipeline(verbose=verbose)
        return self._pipeline

    def translate_to_hindi(self, text: str) -> str:
        """
        Translate English to Hindi using IndicTrans2.
        
        Args:
            text: English text to translate
            
        Returns:
            Hindi translation in Devanagari script
        """
        if not text or not text.strip():
            return ""
        
        # Check cache
        cache_key = f"hi:{text}"
        if cache_key in self._translation_cache:
            logger.debug("Cache hit for Hindi translation")
            return self._translation_cache[cache_key]
        
        try:
            pipeline = self._get_pipeline()
            tokenizer = pipeline['tokenizer']
            model = pipeline['model']
            processor = pipeline['processor']
            device = pipeline['device']
            src_lang = pipeline['src_lang']
            tgt_lang = pipeline['tgt_lang']
            
            # Preprocess with IndicProcessor
            batch = processor.preprocess_batch(
                [text],
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            
            # Tokenize - batch is a list of preprocessed strings
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    use_cache=False,  # Disable cache to avoid None issues
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1
                )
            
            # Decode
            generated_text = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Postprocess with IndicProcessor
            translations = processor.postprocess_batch(generated_text, lang=tgt_lang)
            result = translations[0]
            
            # Cache result
            self._translation_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Hindi translation failed: {str(e)}")
            raise

    def _transliterate_devanagari_to_latin(self, text: str) -> str:
        """
        Transliterate Devanagari to Latin script.
        
        Args:
            text: Text in Devanagari script
            
        Returns:
            Romanized text in Latin script
        """
        result = []
        i = 0
        
        while i < len(text):
            # Check for three-character combinations
            if i < len(text) - 2:
                three_char = text[i:i+3]
                if three_char in DEVANAGARI_TO_LATIN:
                    result.append(DEVANAGARI_TO_LATIN[three_char])
                    i += 3
                    continue
            
            # Check for two-character combinations
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char in DEVANAGARI_TO_LATIN:
                    result.append(DEVANAGARI_TO_LATIN[two_char])
                    i += 2
                    continue
            
            # Single character
            char = text[i]
            if char in DEVANAGARI_TO_LATIN:
                result.append(DEVANAGARI_TO_LATIN[char])
            elif ord(char) >= 0x0900 and ord(char) <= 0x097F:
                # Skip unmapped Devanagari
                pass
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)

    def translate_to_hinglish(self, text: str) -> str:
        """
        Translate to natural Hinglish (romanized Hindi + English in Latin script).
        
        Strategy:
        1. Translate to Hindi using IndicTrans2 (best quality)
        2. Romanize to Latin script
        3. Inject English technical terms for natural code-mixing
        
        Args:
            text: English text to translate
            
        Returns:
            Natural Hinglish in Latin script only
        """
        if not text or not text.strip():
            return ""
        
        # Check cache
        cache_key = f"hinglish:{text}"
        if cache_key in self._translation_cache:
            logger.debug("Cache hit for Hinglish translation")
            return self._translation_cache[cache_key]
        
        try:
            # Get high-quality Hindi translation
            hindi_text = self.translate_to_hindi(text)
            
            # Romanize to Latin script
            romanized = self._transliterate_devanagari_to_latin(hindi_text)
            
            # Inject English technical terms
            hinglish = self._inject_english_terms(romanized, text)
            
            # Clean up
            hinglish = re.sub(r'\s+', ' ', hinglish).strip()
            
            # Cache result
            self._translation_cache[cache_key] = hinglish
            return hinglish
            
        except Exception as e:
            logger.error(f"Hinglish translation failed: {str(e)}")
            raise

    def _inject_english_terms(self, romanized: str, original: str) -> str:
        """
        Inject English technical terms into romanized Hindi.
        
        Args:
            romanized: Romanized Hindi text
            original: Original English text
            
        Returns:
            Hinglish with technical terms in English
        """
        result = romanized
        
        # Pattern matching for common technical terms
        patterns = {
            'email': [r'\beemel\b', r'\bimela\b', r'\beemail\b'],
            'hack': [r'\bhaik\b', r'\bhaika\b'],
            'hacking': [r'\bhaiking\b'],
            'password': [r'\bpaasavarda\b', r'\bpasvarda\b'],
            'account': [r'\bakaaunta\b', r'\bakaunt\b'],
            'system': [r'\bprnaalee\b', r'\bvyvsthaa\b', r'\bsistm\b'],
            'computer': [r'\bknpyutra\b', r'\bkmpyutr\b'],
            'database': [r'\bdetaabes\b', r'\bdetabes\b'],
            'malware': [r'\bmailaveyr\b', r'\bmalveyr\b'],
            'website': [r'\bvebasaaita\b', r'\bvebsait\b'],
            'software': [r'\bsofataveyr\b', r'\bsoftveyr\b'],
            'security': [r'\bsurkshaa\b', r'\bsurksha\b'],
            'network': [r'\bnetavarka\b', r'\bnetvark\b'],
            'server': [r'\bsarvara\b', r'\bsrvr\b'],
            'virus': [r'\bvaayrasa\b', r'\bvayrs\b'],
            'attack': [r'\bhamlaa\b', r'\baakmn\b'],
        }
        
        # Extract words from original
        english_words = original.lower().split()
        
        # Replace romanized versions with English
        for word in english_words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.technical_words and clean_word in patterns:
                for pattern in patterns[clean_word]:
                    result = re.sub(pattern, word, result, flags=re.IGNORECASE)
        
        return result

    def validate_translation(self, original: str, translated: str, target_language: str) -> bool:
        """Validate translation quality."""
        if not translated or not translated.strip():
            return False
        
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', translated))
        has_latin = bool(re.search(r'[a-zA-Z]', translated))
        
        if target_language == 'hindi':
            return has_devanagari
        elif target_language == 'hinglish':
            return has_latin and not has_devanagari
        
        return False

    def batch_translate(self, prompts: List[Prompt], delay: float = 0.05) -> List[Prompt]:
        """
        Batch translate prompts with progress tracking.
        
        Args:
            prompts: List of prompts to translate
            delay: Delay between translations
            
        Returns:
            List of translated prompts
        """
        translated_prompts = []
        total = len(prompts)
        failed_prompts = []
        
        logger.info(f"Starting batch translation of {total} prompts using IndicTrans2...")
        print(f"\n{'='*70}")
        print(f"Batch Translation (AI4Bharat IndicTrans2)")
        print(f"{'='*70}")
        
        # Initialize the translation pipeline before starting batch processing
        if not self.is_initialized():
            try:
                self.initialize_model(verbose=True)
            except Exception as e:
                print(f"✗ Failed to load IndicTrans2: {e}")
                raise
        else:
            print("✓ Using pre-initialized IndicTrans2 model")
        
        print(f"\nStarting translation of {total} prompts...\n")
        start_time = time.time()
        
        for i, prompt in enumerate(prompts, 1):
            progress_pct = (i / total) * 100
            print(f"[{progress_pct:5.1f}%] Prompt {i:3d}/{total} (ID: {prompt.id:20s})...", end=' ', flush=True)
            
            try:
                # Translate
                hindi_text = self.translate_to_hindi(prompt.text_english)
                hinglish_text = self.translate_to_hinglish(prompt.text_english)
                
                # Create translated prompt
                translated_prompt = Prompt(
                    id=prompt.id,
                    category=prompt.category,
                    text_english=prompt.text_english,
                    text_hindi=hindi_text,
                    text_hinglish=hinglish_text,
                    severity=prompt.severity,
                    expected_safe_response=prompt.expected_safe_response
                )
                
                translated_prompts.append(translated_prompt)
                
                # Validate
                hindi_valid = self.validate_translation(prompt.text_english, hindi_text, 'hindi')
                hinglish_valid = self.validate_translation(prompt.text_english, hinglish_text, 'hinglish')
                
                if hindi_valid and hinglish_valid:
                    print("✓")
                else:
                    print("⚠")
                
                if i < total:
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Translation failed for {prompt.id}: {str(e)}")
                print(f"✗")
                
                translated_prompt = Prompt(
                    id=prompt.id,
                    category=prompt.category,
                    text_english=prompt.text_english,
                    text_hindi=None,
                    text_hinglish=None,
                    severity=prompt.severity,
                    expected_safe_response=prompt.expected_safe_response
                )
                translated_prompts.append(translated_prompt)
                failed_prompts.append(prompt.id)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Translation Summary")
        print(f"{'='*70}")
        
        hindi_success = sum(1 for p in translated_prompts if p.text_hindi is not None)
        hinglish_success = sum(1 for p in translated_prompts if p.text_hinglish is not None)
        
        print(f"Total prompts:         {total}")
        print(f"Hindi translations:    {hindi_success}/{total} ({hindi_success/total*100:.1f}%)")
        print(f"Hinglish translations: {hinglish_success}/{total} ({hinglish_success/total*100:.1f}%)")
        print(f"Failed prompts:        {len(failed_prompts)}")
        print(f"Time elapsed:          {elapsed_time:.1f}s")
        print(f"Avg time per prompt:   {elapsed_time/total:.2f}s")
        print(f"Cache entries:         {len(self._translation_cache)}")
        print(f"{'='*70}\n")
        
        return translated_prompts


def translate_dataset() -> None:
    """Convenience function to translate the full dataset."""
    from .prompt_dataset import PromptDataset
    
    dataset = PromptDataset()
    
    try:
        prompts = dataset.load_dataset(str(config.PROMPTS_FILE))
        print(f"Loaded {len(prompts)} prompts from {config.PROMPTS_FILE}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.PROMPTS_FILE}")
        return
    
    engine = TranslationEngine()
    translated_prompts = engine.batch_translate(prompts)
    
    dataset.save_dataset(translated_prompts, str(config.PROMPTS_TRANSLATED_FILE))
    print(f"\n✓ Saved translated dataset to {config.PROMPTS_TRANSLATED_FILE}")


if __name__ == "__main__":
    translate_dataset()
