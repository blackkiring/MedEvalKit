#!/usr/bin/env python
"""
Integration test to verify ToolEvaluator works with MedEvalKit evaluation framework.

This test demonstrates that ToolEvaluator can be used as a drop-in replacement
for models in the existing evaluation pipeline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules directly to avoid torch dependency
import importlib.util

# Load BaseLLM
spec = importlib.util.spec_from_file_location("base_llm", "../models/base_llm.py")
base_llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_llm_module)
BaseLLM = base_llm_module.BaseLLM

# Load ToolEvaluator
spec = importlib.util.spec_from_file_location("tool_evaluator", "../utils/tool_evaluator.py")
tool_eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_eval_module)
ToolEvaluator = tool_eval_module.ToolEvaluator


class MockModel(BaseLLM):
    """Mock model that simulates real model behavior."""
    
    def __init__(self):
        super().__init__()
    
    def process_messages(self, messages):
        return messages
    
    def generate_output(self, messages):
        prompt = messages.get("prompt", "")
        if "<tool_result>" in prompt:
            return f"Answer based on tool result: 42"
        elif "calculate" in prompt.lower():
            return """<tool_call>
{"name": "add", "arguments": {"a": 20, "b": 22}}
</tool_call>"""
        else:
            return "Standard answer without tools"
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def test_basic_functionality():
    """Test basic ToolEvaluator functionality."""
    print("\n" + "="*70)
    print("Test 1: Basic Functionality")
    print("="*70)
    
    model = MockModel()
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model=model, tools=tools)
    
    messages = {"prompt": "What is 20+22? Please calculate."}
    response = evaluator.generate_output(messages)
    
    # Check that tool was called and response contains result
    history = evaluator.get_tool_call_history()
    
    success = len(history) > 0 and "42" in response
    
    if success:
        print(f"✓ Tool called: {len(history)} time(s)")
        print(f"✓ Response contains result: {response[:80]}")
    else:
        print(f"✗ Test failed")
        print(f"  History: {history}")
        print(f"  Response: {response}")
    
    return success

def test_batch_processing():
    """Test batch processing with tools."""
    print("\n" + "="*70)
    print("Test 2: Batch Processing")
    print("="*70)
    
    model = MockModel()
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model=model, tools=tools)
    
    messages_list = [
        {"prompt": "Calculate 20+22"},
        {"prompt": "What is the weather?"},
        {"prompt": "Calculate 10+15"}
    ]
    
    responses = evaluator.generate_outputs(messages_list)
    
    success = len(responses) == 3 and any("42" in r for r in responses)
    
    if success:
        print(f"✓ Processed {len(responses)} messages")
        print(f"✓ At least one response contains tool result")
    else:
        print(f"✗ Test failed")
    
    return success


def test_backward_compatibility():
    """Test that ToolEvaluator maintains backward compatibility."""
    print("\n" + "="*70)
    print("Test 3: Backward Compatibility")
    print("="*70)
    
    base_model = MockModel()
    
    # Create ToolEvaluator with no tools (should behave like base model)
    wrapped_model = ToolEvaluator(
        model=base_model,
        tools={},
        tool_choice="none"
    )
    
    # Test that both produce same output
    messages = {"prompt": "Simple question"}
    
    base_output = base_model.generate_output(messages)
    wrapped_output = wrapped_model.generate_output(messages)
    
    if base_output == wrapped_output:
        print("✓ Backward compatibility verified: Outputs match")
        return True
    else:
        print("✗ Backward compatibility issue: Outputs differ")
        print(f"  Base: {base_output}")
        print(f"  Wrapped: {wrapped_output}")
        return False


def test_interface_compliance():
    """Test that ToolEvaluator implements BaseLLM interface."""
    print("\n" + "="*70)
    print("Test 4: Interface Compliance")
    print("="*70)
    
    base_model = MockModel()
    evaluator = ToolEvaluator(model=base_model)
    
    # Check all required methods exist
    required_methods = ['process_messages', 'generate_output', 'generate_outputs']
    
    all_present = True
    for method in required_methods:
        if hasattr(evaluator, method):
            print(f"✓ Method '{method}' present")
        else:
            print(f"✗ Method '{method}' missing")
            all_present = False
    
    return all_present


def main():
    """Run all integration tests."""
    print("="*70)
    print("MedEvalKit ToolEvaluator Integration Tests")
    print("="*70)
    
    results = {
        "basic_functionality": False,
        "batch_processing": False,
        "backward_compat": False,
        "interface_check": False
    }
    
    try:
        # Test 1: Basic functionality
        results["basic_functionality"] = test_basic_functionality()
        
        # Test 2: Batch processing
        results["batch_processing"] = test_batch_processing()
        
        # Test 3: Backward compatibility
        results["backward_compat"] = test_backward_compatibility()
        
        # Test 4: Interface compliance
        results["interface_check"] = test_interface_compliance()
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ All integration tests passed!")
        print("ToolEvaluator is ready for use with MedEvalKit")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
