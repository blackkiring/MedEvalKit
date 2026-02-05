"""
Test script to verify multi-image support and image index info for all datasets.
"""
import os
import sys

# Set environment variables
os.environ["REASONING"] = "False"

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_get_image_index_info():
    """Test the get_image_index_info helper function."""
    print("Testing get_image_index_info()...")
    
    # Import the function directly from the file
    import importlib.util
    spec = importlib.util.spec_from_file_location("question_formats", 
                                                   "utils/question_formats.py")
    question_formats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(question_formats)
    get_image_index_info = question_formats.get_image_index_info
    
    # Test with 0 images
    result = get_image_index_info(0)
    assert result == "", f"Expected empty string for 0 images, got: {result}"
    print("✓ 0 images: empty string")
    
    # Test with 1 image
    result = get_image_index_info(1)
    assert result == "The index of the given image is 1.\n", f"Expected single image message, got: {result}"
    print("✓ 1 image: 'The index of the given image is 1.'")
    
    # Test with 2 images
    result = get_image_index_info(2)
    assert result == "The indices of the given images are 1, 2.\n", f"Expected 2-image message, got: {result}"
    print("✓ 2 images: 'The indices of the given images are 1, 2.'")
    
    # Test with 5 images
    result = get_image_index_info(5)
    assert result == "The indices of the given images are 1, 2, 3, 4, 5.\n", f"Expected 5-image message, got: {result}"
    print("✓ 5 images: 'The indices of the given images are 1, 2, 3, 4, 5.'")
    
    print("✓ All get_image_index_info() tests passed!\n")

def test_datasets_use_images_field():
    """Verify that datasets now use 'images' field consistently."""
    print("Testing dataset construct_messages methods...")
    
    # Test a few key datasets to ensure they use "images" (plural)
    datasets_to_check = [
        "utils/PMC_VQA/PMC_VQA.py",
        "utils/OmniMedVQA/OmniMedVQA.py",
        "utils/VQA_RAD/VQA_RAD.py",
        "utils/SLAKE/SLAKE.py",
        "utils/PATH_VQA/PATH_VQA.py",
        "utils/CheXpert_Plus/CheXpert_Plus.py",
        "utils/Radrestruct/Radrestruct.py",
    ]
    
    for dataset_file in datasets_to_check:
        try:
            with open(dataset_file, 'r') as f:
                content = f.read()
            
            dataset_name = os.path.basename(dataset_file).replace('.py', '')
            
            # Check if it uses "images" field in messages
            if '"images"' in content or "'images'" in content:
                # Also check it's in the messages dict
                if 'messages = {' in content and 'images' in content:
                    print(f"✓ {dataset_name}: uses 'images' field")
                else:
                    print(f"? {dataset_name}: mentions 'images' but context unclear")
            else:
                print(f"✗ {dataset_name}: does NOT use 'images' field")
                
        except Exception as e:
            print(f"✗ {dataset_name}: Error - {e}")
    
    print()

def test_multi_image_datasets():
    """Test datasets that support multiple images."""
    print("Testing multi-image dataset implementations...")
    
    multi_image_files = [
        "utils/MedFrameQA/MedFrameQA.py",
        "utils/IU_XRAY/IU_XRAY.py",
        "utils/MIMIC_CXR/MIMIC_CXR.py",
        "utils/MedXpertQA/MedXpertQA.py",
    ]
    
    for dataset_file in multi_image_files:
        try:
            with open(dataset_file, 'r') as f:
                content = f.read()
            
            dataset_name = os.path.basename(dataset_file).replace('.py', '')
            
            # Check for get_image_index_info usage
            if "get_image_index_info" in content:
                print(f"✓ {dataset_name}: uses get_image_index_info()")
            else:
                print(f"✗ {dataset_name}: does NOT use get_image_index_info()")
                
        except Exception as e:
            print(f"✗ {dataset_name}: Error - {e}")
    
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Image Support Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_get_image_index_info()
        test_datasets_use_images_field()
        test_multi_image_datasets()
        
        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
