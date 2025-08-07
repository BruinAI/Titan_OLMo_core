#!/usr/bin/env python3
"""
Script to export a padded Dolma2 tokenizer that matches your Titan model's vocab size.

Usage:
    python export_padded_tokenizer.py [OUTPUT_DIR]

If no output directory is specified, it will save to './padded_tokenizer/'
"""

import sys
import os

# Add the src directory to the path so we can import olmo_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.olmo_core.nn.hf.titan_model import export_padded_dolma2_tokenizer

def main():
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "./padded_tokenizer"
    
    print(f"Exporting padded Dolma2 tokenizer to: {output_dir}")
    print("=" * 50)
    
    try:
        export_padded_dolma2_tokenizer(output_dir)
        
        print("\n" + "=" * 50)
        print("SUCCESS! Padded tokenizer exported.")
        print(f"Now copy the tokenizer files from '{output_dir}' to your model directory:")
        print(f"cp {output_dir}/* /path/to/your/titan_hf_model/")
        print("\nOr use the utility directly in your conversion script.")
        
    except Exception as e:
        print(f"ERROR: Failed to export tokenizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
