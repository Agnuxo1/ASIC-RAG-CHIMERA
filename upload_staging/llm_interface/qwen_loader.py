"""
Qwen Model Loader for ASIC-RAG

Handles loading and inference with Qwen3-0.6B model.
Supports:
- Automatic device selection (CPU/CUDA)
- 4-bit quantization for memory efficiency
- Thinking mode for reasoning
- Streaming generation
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator
import warnings

# Suppress warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelConfig:
    """Configuration for Qwen model."""
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    max_context_length: int = 8192
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_4bit: bool = True
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    enable_thinking: bool = True  # Qwen3 thinking mode


class QwenLoader:
    """
    Loader and inference engine for Qwen3 models.
    
    Provides efficient local LLM inference for RAG applications.
    
    Example:
        >>> loader = QwenLoader(ModelConfig())
        >>> response = loader.generate("What is machine learning?")
        >>> print(response)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Qwen loader.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._device = None
        self._loaded = False
        
        # Statistics
        self._total_tokens_generated = 0
        self._total_generation_time = 0.0
        self._inference_count = 0
    
    def load(self) -> bool:
        """
        Load model and tokenizer.
        
        Returns:
            True if successfully loaded
        """
        if self._loaded:
            return True
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model: {self.config.model_name}")
            start_time = time.time()
            
            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            print(f"Using device: {self._device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Configure model loading
            load_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": "auto" if self._device == "cuda" else None,
            }
            
            # Add quantization if using 4-bit
            if self.config.use_4bit and self._device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    load_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print("bitsandbytes not available, loading without quantization")
            
            # Add flash attention if available
            if self.config.use_flash_attention and self._device == "cuda":
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                except:
                    pass
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if self._device == "cpu":
                self.model = self.model.to(self._device)
            
            self.model.eval()
            
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")
            
            self._loaded = True
            return True
            
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Install with: pip install transformers torch accelerate")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation
            enable_thinking: Enable Qwen3 thinking mode
            
        Returns:
            Generated text
        """
        if not self._loaded:
            if not self.load():
                return "Error: Model not loaded"
        
        import torch
        
        start_time = time.time()
        
        # Apply thinking mode if enabled
        thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking
        if thinking:
            prompt = self._apply_thinking_mode(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in response:
                    response = response.split(seq)[0]
        
        # Remove thinking tags if present
        if thinking:
            response = self._remove_thinking_tags(response)
        
        # Update statistics
        gen_time = time.time() - start_time
        self._total_generation_time += gen_time
        self._total_tokens_generated += len(generated_tokens)
        self._inference_count += 1
        
        return response.strip()
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Generated tokens one at a time
        """
        if not self._loaded:
            if not self.load():
                yield "Error: Model not loaded"
                return
        
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for token in streamer:
            yield token
        
        thread.join()
    
    def _apply_thinking_mode(self, prompt: str) -> str:
        """Apply Qwen3 thinking mode to prompt."""
        # Qwen3 uses /think and /no_think tags
        return f"{prompt}\n/think"
    
    def _remove_thinking_tags(self, response: str) -> str:
        """Remove thinking tags from response."""
        import re
        # Remove content between <think> tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt
            
        Returns:
            Assistant response
        """
        # Build prompt from messages
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        return self.generate(prompt)
    
    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        return {
            "model_name": self.config.model_name,
            "device": self._device,
            "loaded": self._loaded,
            "inference_count": self._inference_count,
            "total_tokens_generated": self._total_tokens_generated,
            "total_generation_time": self._total_generation_time,
            "average_tokens_per_second": (
                self._total_tokens_generated / self._total_generation_time
                if self._total_generation_time > 0 else 0.0
            )
        }
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._loaded = False
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


class MockQwenLoader:
    """
    Mock loader for testing without actual model.
    
    Useful for development and testing without GPU.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._loaded = True
    
    def load(self) -> bool:
        return True
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Return mock response based on prompt keywords
        prompt_lower = prompt.lower()
        
        if "revenue" in prompt_lower or "financial" in prompt_lower:
            return "Based on the documents, the Q3 2024 revenue was $45.2 million, representing a 15% increase from the previous quarter."
        elif "keyword" in prompt_lower or "extract" in prompt_lower:
            return "finance, revenue, Q3, 2024, growth, quarterly, report"
        elif "summarize" in prompt_lower:
            return "The document discusses financial performance metrics and quarterly results."
        else:
            return f"This is a mock response to: {prompt[:50]}..."
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        last_message = messages[-1].get("content", "") if messages else ""
        return self.generate(last_message)
    
    def get_statistics(self) -> Dict:
        return {
            "model_name": "MockQwen",
            "device": "mock",
            "loaded": True,
            "inference_count": 0
        }
    
    def unload(self):
        pass


def get_loader(use_mock: bool = False, config: Optional[ModelConfig] = None):
    """
    Factory function to get appropriate loader.
    
    Args:
        use_mock: Use mock loader instead of real model
        config: Model configuration
        
    Returns:
        QwenLoader or MockQwenLoader instance
    """
    if use_mock:
        return MockQwenLoader(config)
    return QwenLoader(config)


if __name__ == "__main__":
    print("Qwen Loader Demo")
    print("=" * 50)
    
    # Use mock for demo
    loader = MockQwenLoader()
    
    # Test generation
    print("\nTest generation:")
    response = loader.generate("What was the Q3 revenue?")
    print(f"Response: {response}")
    
    # Test chat
    print("\nTest chat:")
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "Tell me about the financial report."}
    ]
    response = loader.chat(messages)
    print(f"Response: {response}")
    
    # Statistics
    print("\nStatistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
