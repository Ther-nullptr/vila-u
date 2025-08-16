#!/usr/bin/env python3
"""
Test script for VILA-U MJHQ-30K evaluation pipeline
This script validates the basic functionality without requiring actual models or datasets
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
    from compute_fid_clip_mjhq import (
        load_mjhq_metadata, 
        filter_by_category,
        save_images_vila_u
    )
    from utils import tracker
    print("✅ Successfully imported MJHQ evaluation modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

import numpy as np


def create_mock_mjhq_metadata():
    """Create mock MJHQ metadata for testing"""
    metadata = {
        "1234": {
            "prompt": "A photo of a person wearing a blue shirt",
            "category": ["people"],
            "width": 1024,
            "height": 1024
        },
        "5678": {
            "prompt": "A photo of a golden retriever dog",
            "category": ["animals"],
            "width": 1024,
            "height": 1024
        },
        "9012": {
            "prompt": "A red apple on a wooden table",
            "category": ["objects", "food"],
            "width": 1024,
            "height": 1024
        },
        "3456": {
            "prompt": "A woman walking in a park",
            "category": ["people"],
            "width": 1024,
            "height": 1024
        },
        "7890": {
            "prompt": "A mountain landscape with trees",
            "category": ["landscapes"],
            "width": 1024,
            "height": 1024
        }
    }
    return metadata


def test_metadata_loading():
    """Test MJHQ metadata loading"""
    print("\n=== Testing Metadata Loading ===")
    
    try:
        # Create temporary metadata file
        metadata = create_mock_mjhq_metadata()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f, indent=2)
            temp_file = f.name
        
        # Test loading
        loaded_metadata = load_mjhq_metadata(temp_file)
        
        if loaded_metadata == metadata:
            print("✅ Metadata loading works correctly")
        else:
            print("❌ Metadata loading failed")
            return False
        
        # Cleanup
        os.unlink(temp_file)
        return True
        
    except Exception as e:
        print(f"❌ Error testing metadata loading: {e}")
        return False


def test_category_filtering():
    """Test category filtering functionality"""
    print("\n=== Testing Category Filtering ===")
    
    try:
        metadata = create_mock_mjhq_metadata()
        
        # Test filtering by "people"
        people_data = filter_by_category(metadata, "people")
        expected_people = {"1234", "3456"}  # IDs with "people" category
        
        if set(people_data.keys()) == expected_people:
            print("✅ People category filtering works")
        else:
            print(f"❌ People filtering failed: expected {expected_people}, got {set(people_data.keys())}")
            return False
        
        # Test filtering by "animals"
        animals_data = filter_by_category(metadata, "animals")
        expected_animals = {"5678"}
        
        if set(animals_data.keys()) == expected_animals:
            print("✅ Animals category filtering works")
        else:
            print(f"❌ Animals filtering failed: expected {expected_animals}, got {set(animals_data.keys())}")
            return False
        
        # Test no filtering
        all_data = filter_by_category(metadata, None)
        if len(all_data) == len(metadata):
            print("✅ No filtering (all categories) works")
        else:
            print("❌ No filtering failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing category filtering: {e}")
        return False


def test_image_saving():
    """Test image saving functionality"""
    print("\n=== Testing Image Saving ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock image tensor (similar to VILA-U output)
            # Shape: [batch_size, channels, height, width]
            mock_images = np.random.randint(0, 256, (2, 3, 256, 256), dtype=np.uint8)
            
            # Test saving
            save_images_vila_u(mock_images, temp_dir, "test_img", "people")
            
            # Check if files were created
            expected_files = [
                os.path.join(temp_dir, "people", "test_img_0.png"),
                os.path.join(temp_dir, "people", "test_img_1.png")
            ]
            
            all_exist = all(os.path.exists(f) for f in expected_files)
            
            if all_exist:
                print("✅ Image saving works correctly")
                return True
            else:
                print("❌ Image saving failed - files not created")
                return False
                
    except Exception as e:
        print(f"❌ Error testing image saving: {e}")
        return False


def test_script_structure():
    """Test that required files are in place"""
    print("\n=== Testing Script Structure ===")
    
    required_files = [
        "compute_fid_clip_mjhq.py",
        "run_mjhq_evaluation.sh",
        "utils.py",
        "README_MJHQ.md"
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
    shell_script = current_dir / "run_mjhq_evaluation.sh"
    if shell_script.exists():
        if os.access(shell_script, os.X_OK):
            print("✅ Shell script is executable")
        else:
            print("⚠️  Shell script exists but is not executable")
            print("   Run: chmod +x run_mjhq_evaluation.sh")
    
    return all_present


def test_dependencies():
    """Test that required dependencies are available"""
    print("\n=== Testing Dependencies ===")
    
    dependencies = [
        ("numpy", "np"),
        ("PIL", "PIL"),
        ("tqdm", "tqdm"),
    ]
    
    optional_deps = [
        ("clip", "clip"),
        ("cleanfid", "cleanfid"),
        ("torch", "torch"),
    ]
    
    all_required = True
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {pkg_name} available")
        except ImportError:
            print(f"❌ {pkg_name} missing")
            all_required = False
    
    for pkg_name, import_name in optional_deps:
        try:
            __import__(import_name)
            print(f"✅ {pkg_name} available (optional)")
        except ImportError:
            print(f"⚠️  {pkg_name} missing (optional)")
    
    return all_required


def test_configuration_parsing():
    """Test configuration and argument parsing"""
    print("\n=== Testing Configuration ===")
    
    try:
        # Test sample metadata structure
        metadata = create_mock_mjhq_metadata()
        
        # Validate required fields
        required_fields = ['prompt', 'category']
        
        for img_id, data in metadata.items():
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing field '{field}' in metadata for {img_id}")
                    return False
        
        print("✅ Metadata structure validation passed")
        
        # Test category formats
        for img_id, data in metadata.items():
            category = data['category']
            if not isinstance(category, list):
                print(f"❌ Category should be a list for {img_id}")
                return False
        
        print("✅ Category format validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False


def main():
    """Run all tests"""
    print("VILA-U MJHQ-30K Evaluation Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Script Structure", test_script_structure),
        ("Dependencies", test_dependencies),
        ("Metadata Loading", test_metadata_loading),
        ("Category Filtering", test_category_filtering),
        ("Image Saving", test_image_saving),
        ("Configuration", test_configuration_parsing),
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
        print("🎉 All tests passed! MJHQ pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Download MJHQ-30K dataset from Hugging Face")
        print("2. Install optional dependencies: pip install clip-by-openai cleanfid")
        print("3. Run evaluation: ./run_mjhq_evaluation.sh --model-path /path/to/vila-u/model")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())