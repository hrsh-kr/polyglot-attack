"""
Ollama Interface Module

Handles communication with the local Llama model via Ollama API.
Records responses with complete metadata for reproducibility.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import time

try:
    import ollama
except ImportError:
    raise ImportError(
        "ollama package not found. Please install it with: pip install ollama"
    )

import config


@dataclass
class Response:
    """
    Data model for a model response with complete metadata.
    
    Attributes:
        prompt_id: ID of the prompt that generated this response
        prompt_text: The actual prompt text sent to the model
        response_text: The model's response
        timestamp: When the response was generated
        model_name: Name of the model used
        inference_params: Dictionary of inference parameters used
    """
    prompt_id: str
    prompt_text: str
    response_text: str
    timestamp: str
    model_name: str
    inference_params: Dict

    def to_dict(self) -> Dict:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Response':
        """Create Response instance from dictionary."""
        return cls(**data)


class OllamaInterface:
    """
    Interface for communicating with Ollama API.
    
    Provides methods for generating responses with fixed inference parameters
    to ensure reproducibility across experiments.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Ollama interface with configuration.
        
        Args:
            model_name: Name of the Ollama model (default from config)
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            top_p: Top-p sampling parameter (default from config)
            base_url: Ollama API base URL (default from config)
        """
        # Load from config if not provided
        self.model_name = model_name or config.OLLAMA_CONFIG["model_name"]
        self.temperature = temperature if temperature is not None else config.OLLAMA_CONFIG["temperature"]
        self.max_tokens = max_tokens or config.OLLAMA_CONFIG["max_tokens"]
        self.top_p = top_p if top_p is not None else config.OLLAMA_CONFIG["top_p"]
        self.base_url = base_url or config.OLLAMA_CONFIG["base_url"]
        
        # Set up Ollama client
        self.client = ollama.Client(host=self.base_url)
        
        # Store inference parameters for metadata
        self.inference_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def generate_response(self, prompt: str, prompt_id: str = "unknown") -> Response:
        """
        Generate a response for a single prompt with fixed inference parameters.
        
        Args:
            prompt: The prompt text to send to the model
            prompt_id: Identifier for the prompt (for tracking)
            
        Returns:
            Response object with complete metadata
            
        Raises:
            Exception: If Ollama API call fails
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Call Ollama API with fixed parameters
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": self.top_p,
                }
            )
            
            # Extract response text
            response_text = response.get("response", "")
            
            # Create Response object
            return Response(
                prompt_id=prompt_id,
                prompt_text=prompt,
                response_text=response_text,
                timestamp=timestamp,
                model_name=self.model_name,
                inference_params=self.inference_params.copy()
            )
            
        except Exception as e:
            # Return error response
            return Response(
                prompt_id=prompt_id,
                prompt_text=prompt,
                response_text=f"ERROR: {str(e)}",
                timestamp=timestamp,
                model_name=self.model_name,
                inference_params=self.inference_params.copy()
            )

    def batch_generate(
        self,
        prompts: List[tuple],
        retry_attempts: Optional[int] = None,
        retry_backoff: Optional[float] = None
    ) -> List[Response]:
        """
        Generate responses for multiple prompts with retry logic.
        
        Args:
            prompts: List of (prompt_text, prompt_id) tuples
            retry_attempts: Number of retry attempts for failed requests
            retry_backoff: Exponential backoff multiplier for retries
            
        Returns:
            List of Response objects
        """
        retry_attempts = retry_attempts or config.RED_TEAM_CONFIG["retry_attempts"]
        retry_backoff = retry_backoff or config.RED_TEAM_CONFIG["retry_backoff"]
        
        responses = []
        
        for prompt_text, prompt_id in prompts:
            response = None
            
            # Retry logic
            for attempt in range(retry_attempts):
                try:
                    response = self.generate_response(prompt_text, prompt_id)
                    
                    # Check if response is an error
                    if not response.response_text.startswith("ERROR:"):
                        break
                    
                    # If error and not last attempt, wait and retry
                    if attempt < retry_attempts - 1:
                        wait_time = retry_backoff ** attempt
                        time.sleep(wait_time)
                        
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = retry_backoff ** attempt
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed, create error response
                        response = Response(
                            prompt_id=prompt_id,
                            prompt_text=prompt_text,
                            response_text=f"ERROR: All retry attempts failed. Last error: {str(e)}",
                            timestamp=datetime.now().isoformat(),
                            model_name=self.model_name,
                            inference_params=self.inference_params.copy()
                        )
            
            if response:
                responses.append(response)
        
        return responses

    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information (JSON serializable)
        """
        try:
            # Get model details from Ollama
            model_info = self.client.show(self.model_name)
            
            # Convert ShowResponse (Pydantic model) to dict
            if hasattr(model_info, 'model_dump'):
                # Pydantic v2
                details = model_info.model_dump()
            elif hasattr(model_info, 'dict'):
                # Pydantic v1
                details = model_info.dict()
            elif isinstance(model_info, dict):
                details = model_info
            else:
                details = {"raw": str(model_info)}
            
            # Convert datetime objects to ISO format strings for JSON serialization
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return obj
            
            details = make_json_serializable(details)
            
            return {
                "model_name": self.model_name,
                "details": details,
                "inference_params": self.inference_params.copy()
            }
        except Exception as e:
            return {
                "model_name": self.model_name,
                "error": str(e),
                "inference_params": self.inference_params.copy()
            }

    def save_responses(self, responses: List[Response], filepath: str) -> None:
        """
        Save responses to JSON file.
        
        Args:
            responses: List of Response objects
            filepath: Path to save JSON file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [response.to_dict() for response in responses]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_responses(self, filepath: str) -> List[Response]:
        """
        Load responses from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of Response objects
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Responses file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [Response.from_dict(item) for item in data]


def test_ollama_connection() -> bool:
    """
    Test connection to Ollama and verify model is available.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        interface = OllamaInterface()
        model_info = interface.get_model_info()
        
        if "error" in model_info:
            print(f"✗ Error connecting to Ollama: {model_info['error']}")
            return False
        
        print(f"✓ Successfully connected to Ollama")
        print(f"✓ Model: {interface.model_name}")
        print(f"✓ Inference parameters: {interface.inference_params}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the connection
    test_ollama_connection()

