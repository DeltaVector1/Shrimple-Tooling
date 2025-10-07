#!/usr/bin/env python3
"""
Script to remove '._checkpoint_wrapped_module' from safetensors file keys
and update the corresponding index.json file.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install with: pip install safetensors torch")
    sys.exit(1)


def clean_safetensors_file(file_path: str) -> bool:
    """
    Remove '._checkpoint_wrapped_module' from keys in a safetensors file.
    
    Args:
        file_path: Path to the safetensors file
        
    Returns:
        bool: True if any keys were modified, False otherwise
    """
    print(f"Processing safetensors file: {file_path}")
    
    # Load the safetensors file
    tensors = {}
    metadata = {}
    keys_modified = False
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Get metadata
            metadata = f.metadata() or {}
            
            # Load all tensors with cleaned keys
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Check if key contains the unwanted prefix
                if "._checkpoint_wrapped_module" in key:
                    new_key = key.replace("._checkpoint_wrapped_module", "")
                    print(f"  Renaming: {key} -> {new_key}")
                    tensors[new_key] = tensor
                    keys_modified = True
                else:
                    tensors[key] = tensor
    
    except Exception as e:
        print(f"Error reading safetensors file: {e}")
        return False
    
    if not keys_modified:
        print("  No keys needed modification.")
        return False
    
    # Save the modified tensors back to the same file
    try:
        save_file(tensors, file_path, metadata=metadata)
        print(f"  Successfully updated: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving safetensors file: {e}")
        return False


def update_index_json(index_path: str) -> bool:
    """
    Update the index.json file to reflect the cleaned keys.
    
    Args:
        index_path: Path to the index.json file
        
    Returns:
        bool: True if file was modified, False otherwise
    """
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return False
    
    print(f"Processing index file: {index_path}")
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    except Exception as e:
        print(f"Error reading index file: {e}")
        return False
    
    modified = False
    
    # Update weight_map if it exists
    if 'weight_map' in index_data:
        new_weight_map = {}
        for key, filename in index_data['weight_map'].items():
            if "._checkpoint_wrapped_module" in key:
                new_key = key.replace("._checkpoint_wrapped_module", "")
                print(f"  Index: {key} -> {new_key}")
                new_weight_map[new_key] = filename
                modified = True
            else:
                new_weight_map[key] = filename
        index_data['weight_map'] = new_weight_map
    
    if not modified:
        print("  No index entries needed modification.")
        return False
    
    # Save the updated index
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"  Successfully updated: {index_path}")
        return True
    except Exception as e:
        print(f"Error saving index file: {e}")
        return False


def process_directory(directory: str) -> None:
    """
    Process all safetensors files in a directory and update index.json if present.
    
    Args:
        directory: Path to the directory containing safetensors files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory does not exist: {directory}")
        return
    
    # Find all safetensors files
    safetensors_files = list(dir_path.glob("*.safetensors"))
    
    if not safetensors_files:
        print(f"No safetensors files found in: {directory}")
        return
    
    print(f"Found {len(safetensors_files)} safetensors file(s)")
    
    # Process each safetensors file
    any_modified = False
    for file_path in safetensors_files:
        if clean_safetensors_file(str(file_path)):
            any_modified = True
    
    # Update index.json if it exists and any files were modified
    index_path = dir_path / "model.index.json"
    if any_modified and index_path.exists():
        update_index_json(str(index_path))
    elif any_modified:
        print("Note: No model.index.json found to update")


def main():
    parser = argparse.ArgumentParser(
        description="Remove '._checkpoint_wrapped_module' from safetensors file keys"
    )
    parser.add_argument(
        "path",
        help="Path to safetensors file or directory containing safetensors files"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    if path.is_file() and path.suffix == ".safetensors":
        # Process single file
        clean_safetensors_file(str(path))
        
        # Check for index file in the same directory
        index_path = path.parent / "model.index.json"
        if index_path.exists():
            update_index_json(str(index_path))
    
    elif path.is_dir():
        # Process directory
        process_directory(str(path))
    
    else:
        print(f"Error: Path must be a .safetensors file or directory: {args.path}")
        sys.exit(1)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
