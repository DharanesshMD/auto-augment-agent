#!/usr/bin/env python3
"""Test a trained model or baseline.

Usage:
    python scripts/test_model.py --model-path outputs/baseline/model
    python scripts/test_model.py --model-path outputs/trial_5/model --input "The weather is"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config

console = Console()
logger = logging.getLogger(__name__)


def test_nlp(model, tokenizer, input_text, device):
    """Run inference for NLP tasks."""
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if hasattr(model, "generate"):
            # Causal LM
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Classification
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            return {"label": pred, "probability": probs[0][pred].item()}


def main():
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--input", type=str, default=None, help="Input text for NLP tasks")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    
    from src.utils.reproducibility import get_device
    device = get_device(args.device)
    
    config = load_config()
    task = config["model"]["task"]
    model_name = config["model"]["name"]
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model path {model_path} does not exist.")
        return

    console.print(f"Loading model from [cyan]{model_path}[/cyan]...")
    
    if task in ("language_modeling", "text_classification"):
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        if task == "language_modeling":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            
        # Load LoRA or full weights
        if (model_path / "adapter_config.json").exists():
            from src.training.lora import load_lora_weights
            model = load_lora_weights(model, model_path)
        elif (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            # If it was a full save_pretrained
            if task == "language_modeling":
                model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        
        # Inference
        input_text = args.input or "The model is now"
        console.print(f"\n[bold]Input:[/bold] {input_text}")
        result = test_nlp(model, tokenizer, input_text, device)
        console.print(f"[bold green]Output:[/bold green]\n{result}")

    elif task == "image_classification":
        console.print("Image classification inference not fully implemented in CLI yet.")
    
    else:
        console.print(f"Testing not yet implemented for task: {task}")


if __name__ == "__main__":
    main()
