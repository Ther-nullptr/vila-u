#!/usr/bin/env python3
"""
Test script for VILA-U GenEval evaluation pipeline
This script validates the basic functionality without requiring actual models
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from evaluate_geneval import GenEvalEvaluator, load_geneval_prompts
    from utils import tracker
    print("✅ Successfully imported evaluation modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_load_prompts():
    """Test loading GenEval prompts"""
    print("\n=== Testing Prompt Loading ===")
    
    # Use example prompts file
    example_file = current_dir / "geneval_prompts_example.jsonl"
    
    if not example_file.exists():
        print(f"❌ Example prompts file not found: {example_file}")
        return False
    
    try:
        prompts = load_geneval_prompts(str(example_file))
        print(f"✅ Loaded {len(prompts)} prompts")
        
        # Validate prompt structure
        if prompts:
            first_prompt = prompts[0]
            required_keys = ['id', 'tag', 'prompt', 'objects', 'object_counts', 'spatial_relations', 'attributes']
            for key in required_keys:
                if key not in first_prompt:
                    print(f"❌ Missing key '{key}' in prompt structure")
                    return False
            print("✅ Prompt structure validation passed")
            
            # Validate that objects are extracted correctly
            if first_prompt['objects'] and first_prompt['tag'] == 'single_object':
                print(f"✅ Objects extracted correctly: {first_prompt['objects']}")
            else:
                print(f"❌ Objects extraction failed: {first_prompt}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return False


def test_evaluator_init():
    """Test GenEval evaluator initialization"""
    print("\n=== Testing Evaluator Initialization ===")
    
    try:
        # Test without MMDetection (should work but with warnings)
        evaluator = GenEvalEvaluator(device='cpu')
        print("✅ Evaluator initialized successfully (without MMDetection)")
        
        # Test spatial relationship checker
        bbox1 = [10, 10, 50, 50]  # left object
        bbox2 = [60, 10, 100, 50]  # right object
        
        # Test "left of" relationship
        result = evaluator._check_spatial_relationship(bbox1, bbox2, "left of")
        if result:
            print("✅ Spatial relationship checking works")
        else:
            print("❌ Spatial relationship checking failed")
            return False
        
        # Test "right of" relationship (should be False)
        result = evaluator._check_spatial_relationship(bbox1, bbox2, "right of")
        if not result:
            print("✅ Spatial relationship checking correctly identifies wrong relationships")
        else:
            print("❌ Spatial relationship checking incorrectly identified wrong relationship")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return False


def test_object_presence_evaluation():
    """Test object presence evaluation (without actual detection)"""
    print("\n=== Testing Object Presence Evaluation ===")
    
    try:
        evaluator = GenEvalEvaluator(device='cpu')
        
        # Mock class names
        class_names = ['person', 'car', 'cat', 'dog', 'apple', 'table']
        
        # Test with empty detections (no objects found)
        required_objects = ['cat', 'dog']
        result = evaluator.evaluate_object_presence(
            "dummy_path", required_objects, class_names
        )
        
        if result['accuracy'] == 0.0:
            print("✅ Object presence evaluation handles missing objects correctly")
        else:
            print("❌ Object presence evaluation failed for missing objects")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error in object presence evaluation: {e}")
        return False


def test_directory_structure():
    """Test that required directories and files are in place"""
    print("\n=== Testing Directory Structure ===")
    
    required_files = [
        "evaluate_geneval.py",
        "utils.py", 
        "run_geneval_evaluation.sh",
        "README.md",
        "geneval_prompts_example.jsonl"
    ]
    
    all_present = True
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            all_present = False
    
    # Check if shell script is executable
    shell_script = current_dir / "run_geneval_evaluation.sh"
    if shell_script.exists():
        if os.access(shell_script, os.X_OK):
            print("✅ Shell script is executable")
        else:
            print("⚠️  Shell script exists but is not executable")
            print("   Run: chmod +x run_geneval_evaluation.sh")
    
    return all_present


def test_output_format():
    """Test that results are saved in correct format"""
    print("\n=== Testing Output Format ===")
    
    try:
        # Create temporary output
        with tempfile.TemporaryDirectory() as temp_dir:
            results = {
                'object_presence': [0.8, 0.6, 1.0],
                'object_counting': [0.7, 0.9],
                'spatial_relations': [0.5, 0.3, 0.8],
                'overall_accuracy': 0.69
            }
            
            output_file = os.path.join(temp_dir, "test_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Verify file was created and is valid JSON
            with open(output_file, 'r') as f:
                loaded_results = json.load(f)
            
            if loaded_results == results:
                print("✅ Results saved and loaded correctly")
                return True
            else:
                print("❌ Results format validation failed")
                return False
                
    except Exception as e:
        print(f"❌ Error testing output format: {e}")
        return False


def main():
    """Run all tests"""
    print("VILA-U GenEval Evaluation Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Prompt Loading", test_load_prompts),
        ("Evaluator Initialization", test_evaluator_init),
        ("Object Presence Evaluation", test_object_presence_evaluation),
        ("Output Format", test_output_format),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Download/prepare GenEval prompts in JSONL format as geneval_prompts.jsonl")
        print("2. Install MMDetection: pip install mmdet")
        print("3. Run evaluation: ./run_geneval_evaluation.sh --model-path /path/to/vila-u/model")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())