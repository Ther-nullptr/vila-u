#!/usr/bin/env python3
"""
Demo script to show the expected folder structure for VILA-U GenEval evaluation
Creates sample folders without actually generating images
"""

import json
import os
import tempfile
from pathlib import Path

def create_demo_structure(output_dir="demo_output"):
    """Create demo folder structure for GenEval evaluation"""
    
    print(f"Creating demo folder structure in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample prompts
    sample_prompts = [
        {"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "a photo of a cow"},
        {"tag": "two_object", "include": [{"class": "cat", "count": 1}, {"class": "dog", "count": 1}], "prompt": "a photo of a cat and a dog"},
        {"tag": "counting", "include": [{"class": "apple", "count": 3}], "prompt": "a photo of three apples"},
    ]
    
    for prompt_idx, prompt_data in enumerate(sample_prompts):
        # Create prompt folder: 00000/, 00001/, etc.
        prompt_folder = os.path.join(output_dir, f"{prompt_idx:05d}")
        samples_folder = os.path.join(prompt_folder, "samples")
        os.makedirs(samples_folder, exist_ok=True)
        
        # Create metadata.jsonl
        metadata_path = os.path.join(prompt_folder, "metadata.jsonl")
        with open(metadata_path, 'w') as f:
            json.dump(prompt_data, f)
            f.write('\n')
        
        # Create placeholder images (empty files for demo)
        for sample_idx in range(2):  # 2 samples per prompt
            sample_path = os.path.join(samples_folder, f"{sample_idx:04d}.png")
            # Create empty placeholder file
            with open(sample_path, 'w') as f:
                f.write(f"# Placeholder for image {sample_idx} of prompt {prompt_idx}\n")
        
        print(f"Created: {prompt_folder}/")
        print(f"  ├── metadata.jsonl")
        print(f"  └── samples/")
        print(f"      ├── 0000.png")
        print(f"      └── 0001.png")
    
    print(f"\n✅ Demo structure created in: {output_dir}")
    print(f"\nExpected structure:")
    print(f"{output_dir}/")
    for i in range(len(sample_prompts)):
        print(f"├── {i:05d}/")
        print(f"│   ├── metadata.jsonl")
        print(f"│   └── samples/")
        print(f"│       ├── 0000.png")
        print(f"│       └── 0001.png")
    
    return output_dir

def validate_structure(base_dir):
    """Validate that the folder structure matches requirements"""
    print(f"\n=== Validating structure in {base_dir} ===")
    
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return False
    
    prompt_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    prompt_folders.sort()
    
    if not prompt_folders:
        print("❌ No prompt folders found")
        return False
    
    print(f"✅ Found {len(prompt_folders)} prompt folders")
    
    for folder in prompt_folders:
        folder_path = os.path.join(base_dir, folder)
        
        # Check metadata.jsonl
        metadata_path = os.path.join(folder_path, "metadata.jsonl")
        if os.path.exists(metadata_path):
            print(f"✅ {folder}/metadata.jsonl exists")
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    if 'tag' in data and 'prompt' in data:
                        print(f"  📝 Valid metadata: {data['tag']} - {data['prompt'][:50]}...")
                    else:
                        print(f"  ⚠️  Metadata missing required fields")
            except:
                print(f"  ❌ Invalid metadata format")
        else:
            print(f"❌ {folder}/metadata.jsonl missing")
        
        # Check samples folder
        samples_path = os.path.join(folder_path, "samples")
        if os.path.exists(samples_path):
            samples = [f for f in os.listdir(samples_path) if f.endswith('.png')]
            print(f"✅ {folder}/samples/ exists with {len(samples)} images")
        else:
            print(f"❌ {folder}/samples/ missing")
    
    print("\n✅ Structure validation complete")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo VILA-U GenEval folder structure")
    parser.add_argument("--output-dir", default="demo_output", help="Output directory")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing structure")
    parser.add_argument("--cleanup", action="store_true", help="Remove demo folder after validation")
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_structure(args.output_dir)
    else:
        # Create demo structure
        output_dir = create_demo_structure(args.output_dir)
        
        # Validate it
        validate_structure(output_dir)
        
        if args.cleanup:
            import shutil
            shutil.rmtree(output_dir)
            print(f"\n🧹 Cleaned up demo folder: {output_dir}")
        else:
            print(f"\n💡 To clean up: rm -rf {output_dir}")
            print(f"💡 To validate: python {__file__} --validate-only --output-dir {output_dir}")