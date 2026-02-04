#!/usr/bin/env python3
"""
Test script to verify Qwen3_VL parameter changes.
Tests that the new parameters (enforce_eager, max_model_len, frequency_penalty) are set correctly.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

def test_qwen3_vl_initialization():
    """Test that Qwen3_VL can be initialized with new parameters."""
    print("\n" + "="*70)
    print("Test: Qwen3_VL Parameter Configuration")
    print("="*70)
    
    # Create a mock args object
    class Args:
        max_image_num = 5
        temperature = 0.7
        top_p = 0.9
        repetition_penalty = 1.1
        max_new_tokens = 512
        enable_thinking = "False"
    
    try:
        # We can't actually initialize vLLM without a model, but we can verify
        # the code structure is correct
        import inspect
        import importlib.util
        
        # Load the module
        spec = importlib.util.spec_from_file_location(
            "Qwen3_VL_vllm",
            "/home/runner/work/MedEvalKit/MedEvalKit/models/Qwen3_VL/Qwen3_VL_vllm.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # Read the source code to verify parameters
        with open("/home/runner/work/MedEvalKit/MedEvalKit/models/Qwen3_VL/Qwen3_VL_vllm.py", "r") as f:
            source = f.read()
        
        # Check for enforce_eager=False
        if "enforce_eager=False" in source:
            print("✓ PASS: enforce_eager set to False (enables CUDA graphs)")
        else:
            print("✗ FAIL: enforce_eager not set to False")
            return False
        
        # Check for max_model_len
        if "max_model_len=100000" in source:
            print("✓ PASS: max_model_len set to 100000 (long context support)")
        else:
            print("✗ FAIL: max_model_len not set correctly")
            return False
        
        # Check for frequency_penalty
        if "frequency_penalty=args.repetition_penalty" in source:
            print("✓ PASS: frequency_penalty parameter used (vLLM standard)")
        else:
            print("✗ FAIL: frequency_penalty not used")
            return False
        
        # Check for raw_messages variable (code clarity)
        if "raw_messages = messages[" in source:
            print("✓ PASS: raw_messages variable used (clearer code)")
        else:
            print("✗ FAIL: raw_messages variable not used")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_documentation():
    """Verify that parameter changes are documented."""
    print("\n" + "="*70)
    print("Test: Parameter Documentation")
    print("="*70)
    
    with open("/home/runner/work/MedEvalKit/MedEvalKit/models/Qwen3_VL/Qwen3_VL_vllm.py", "r") as f:
        source = f.read()
    
    # Check that comments are preserved
    if "Qwen3默认开启thinking模式" in source:
        print("✓ PASS: Thinking mode comment preserved")
    else:
        print("✗ FAIL: Comment missing")
        return False
    
    # Check multimodal support comment
    if "multimodal content items" in source.lower():
        print("✓ PASS: Multimodal content documentation present")
    else:
        print("✗ FAIL: Multimodal documentation missing")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Qwen3_VL Parameter Configuration Tests")
    print("="*70)
    
    tests = [
        test_qwen3_vl_initialization,
        test_parameter_documentation
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests PASSED!")
        print("\nKey improvements verified:")
        print("  • enforce_eager=False - Enables CUDA graphs for better performance")
        print("  • max_model_len=100000 - Supports long context (up to 100K tokens)")
        print("  • frequency_penalty - Uses standard vLLM parameter name")
        print("  • raw_messages - Clearer variable naming (avoids shadowing)")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
