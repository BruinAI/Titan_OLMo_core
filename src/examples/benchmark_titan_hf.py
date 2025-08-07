"""
Example of how to use the converted Titan HuggingFace model for benchmarking.

This shows how your converted model can be used with standard HuggingFace tools
while preserving all the custom Titan functionality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

def load_converted_titan_model(model_path: str):
    """
    Load your converted Titan model from HuggingFace format.
    
    Args:
        model_path: Path to the converted HuggingFace model directory
    """
    
    # Important: use trust_remote_code=True to load custom model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use half precision if needed
        device_map="auto"  # Automatically handle device placement
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def run_benchmark_example(model_path: str):
    """
    Example of running a standardized benchmark with your Titan model.
    """
    
    # Load the model
    model, tokenizer = load_converted_titan_model(model_path)
    
    # Example 1: Use with HuggingFace pipeline
    print("=== Pipeline Example ===")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8
    )
    
    prompt = "The future of artificial intelligence is"
    outputs = generator(prompt)
    print(f"Input: {prompt}")
    print(f"Output: {outputs[0]['generated_text']}")
    
    # Example 2: Direct model usage
    print("\n=== Direct Model Usage ===")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # Example 3: Access the underlying Titan model for custom operations
    print("\n=== Accessing Titan Components ===")
    titan_model = model.model  # This is your original Titan model
    
    # Check if memory modules are present
    memory_modules = []
    for name, module in titan_model.named_modules():
        if hasattr(module, 'memory') and module.memory is not None:
            memory_modules.append(name)
    
    print(f"Found {len(memory_modules)} memory modules:")
    for name in memory_modules:
        print(f"  - {name}")
    
    return model, tokenizer


def run_evaluation_suite(model_path: str):
    """
    Example of how to integrate with evaluation frameworks.
    
    This shows how your model can work with libraries like:
    - lm-evaluation-harness
    - EleutherAI evaluation suite
    - Custom benchmark suites
    """
    
    model, tokenizer = load_converted_titan_model(model_path)
    
    # Example integration with evaluation framework
    # This is pseudocode - adapt to your specific evaluation needs
    
    print("=== Evaluation Example ===")
    
    # The key point is that your model now works with any HuggingFace-compatible
    # evaluation framework while preserving all Titan functionality
    
    # Example evaluation tasks:
    eval_tasks = [
        "hellaswag",
        "winogrande", 
        "arc_easy",
        "arc_challenge",
        "boolq"
    ]
    
    print(f"Model can be evaluated on: {eval_tasks}")
    print("Use standard evaluation frameworks like lm-evaluation-harness")
    print("Example command:")
    print(f"  lm_eval --model hf --model_args pretrained={model_path},trust_remote_code=True --tasks {','.join(eval_tasks)}")


"""
MIGRATION TO DIFFERENT REPOSITORY FOR BENCHMARKING:

To use your converted Titan model in a completely different repository/environment:

1. **What you need to bring:**
   - The entire converted model directory (contains config.json, pytorch_model.bin, etc.)
   - The modeling_titan_olmo.py file (created automatically by convert_titan_to_hf.py)
   - tokenizer files (tokenizer.json, etc.)

2. **Dependencies in the new repository:**
   You need to install the olmo_core package or ensure these dependencies are available:
   - olmo_core.nn.transformer.model
   - olmo_core.nn.titans.neural_memory  
   - olmo_core.nn.transformer.config
   - olmo_core.memory_config
   - olmo_core.config (for DType)

3. **Loading in the new repository:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   # This will automatically use the modeling_titan_olmo.py file
   model = AutoModelForCausalLM.from_pretrained(
       "path/to/converted/model",
       trust_remote_code=True,  # Essential - allows loading custom model code
       torch_dtype=torch.bfloat16
   )
   tokenizer = AutoTokenizer.from_pretrained("path/to/converted/model")
   ```

4. **What trust_remote_code=True does:**
   - Loads the modeling_titan_olmo.py file from the model directory
   - Executes the custom TitanOLMoForCausalLM and TitanOLMoConfig classes
   - Preserves all your custom Titan functionality

5. **Alternative approach (more secure):**
   If you don't want to use trust_remote_code=True, you can:
   - Copy the TitanOLMoConfig and TitanOLMoForCausalLM classes to your new repo
   - Import them explicitly and load the model manually

6. **For evaluation frameworks:**
   Most evaluation frameworks (like lm-evaluation-harness) support trust_remote_code=True:
   ```bash
   lm_eval --model hf --model_args pretrained=path/to/model,trust_remote_code=True --tasks hellaswag,arc_easy
   ```

The key insight is that the HuggingFace format packages everything needed to reconstruct 
your model, including the complete transformer configuration and all memory settings.
"""

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python benchmark_example.py <path_to_converted_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        model, tokenizer = run_benchmark_example(model_path)
        run_evaluation_suite(model_path)
        
        print("\n=== Success! ===")
        print("Your Titan model is now ready for standardized benchmarking!")
        print("All custom neural memory functionality is preserved.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've converted your model first using convert_titan_to_hf.py")
