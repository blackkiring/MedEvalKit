#!/usr/bin/env python3
"""
Test script to verify ToolEvaluator correctly injects medical system prompts.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from tool_evaluator import ToolEvaluator, MEDICAL_AGENT_SYSTEM_PROMPT
from base_llm import BaseLLM


class MockModel(BaseLLM):
    """Mock model for testing that captures the system prompt."""
    
    def __init__(self):
        super().__init__()
        self.last_messages = None
    
    def process_messages(self, messages):
        self.last_messages = messages
        return messages
    
    def generate_output(self, messages):
        self.last_messages = messages
        # Return a response without tool calls
        return "This is a test response."
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def test_system_prompt_injection_with_medical_tools():
    """Test that system prompt is injected when medical_tools_config is provided."""
    print("\n" + "="*70)
    print("Test 1: System prompt injection with medical tools")
    print("="*70)
    
    model = MockModel()
    medical_config = {
        "output_dir": "/tmp/test_medical_outputs"
    }
    
    evaluator = ToolEvaluator(
        model=model,
        medical_tools_config=medical_config
    )
    
    # Test messages without system prompt
    messages = {"prompt": "Analyze this medical image."}
    
    # Generate output
    result = evaluator.generate_output(messages)
    
    # Check if system prompt was injected
    if model.last_messages is not None and "system" in model.last_messages:
        print("✓ PASS: System prompt was injected")
        print(f"  System prompt length: {len(model.last_messages['system'])} characters")
        print(f"  System prompt starts with: {model.last_messages['system'][:80]}...")
        
        # Verify it's the medical agent prompt
        if "medical image analysis" in model.last_messages['system'].lower():
            print("✓ PASS: Correct medical system prompt injected")
            return True
        else:
            print("✗ FAIL: Wrong system prompt content")
            return False
    else:
        print("✗ FAIL: System prompt was not injected")
        return False


def test_no_injection_without_medical_tools():
    """Test that system prompt is NOT injected when medical_tools_config is not provided."""
    print("\n" + "="*70)
    print("Test 2: No system prompt injection without medical tools")
    print("="*70)
    
    model = MockModel()
    
    evaluator = ToolEvaluator(model=model)
    
    # Test messages without system prompt
    messages = {"prompt": "Analyze this image."}
    
    # Generate output
    result = evaluator.generate_output(messages)
    
    # Check if system prompt was NOT injected
    if model.last_messages is not None and "system" not in model.last_messages:
        print("✓ PASS: System prompt was not injected (as expected)")
        return True
    else:
        print("✗ FAIL: System prompt was injected unexpectedly")
        return False


def test_preserve_existing_system_prompt():
    """Test that existing system prompt is preserved and not overwritten."""
    print("\n" + "="*70)
    print("Test 3: Preserve existing system prompt")
    print("="*70)
    
    model = MockModel()
    medical_config = {
        "output_dir": "/tmp/test_medical_outputs"
    }
    
    evaluator = ToolEvaluator(
        model=model,
        medical_tools_config=medical_config
    )
    
    # Test messages WITH system prompt
    custom_prompt = "This is a custom system prompt."
    messages = {
        "prompt": "Analyze this image.",
        "system": custom_prompt
    }
    
    # Generate output
    result = evaluator.generate_output(messages)
    
    # Check if original system prompt was preserved
    if model.last_messages is not None and "system" in model.last_messages:
        if model.last_messages["system"] == custom_prompt:
            print("✓ PASS: Existing system prompt was preserved")
            return True
        else:
            print("✗ FAIL: Existing system prompt was overwritten")
            print(f"  Expected: {custom_prompt}")
            print(f"  Got: {model.last_messages['system']}")
            return False
    else:
        print("✗ FAIL: System prompt was lost")
        return False


def test_prompt_content_includes_all_sections():
    """Test that the injected prompt includes all required sections."""
    print("\n" + "="*70)
    print("Test 4: Prompt content includes all required sections")
    print("="*70)
    
    model = MockModel()
    medical_config = {
        "output_dir": "/tmp/test_medical_outputs"
    }
    
    evaluator = ToolEvaluator(
        model=model,
        medical_tools_config=medical_config
    )
    
    # Generate output
    messages = {"prompt": "Test"}
    result = evaluator.generate_output(messages)
    
    if model.last_messages is None or "system" not in model.last_messages:
        print("✗ FAIL: No system prompt found")
        return False
    
    system_prompt = model.last_messages["system"]
    
    # Check for required sections
    required_sections = [
        "Available tools:",
        "Zoom-in",
        "BiomedParse",
        "SAM2",
        "Required Output Format:",
        "reasoning techniques",
        "medical image analysis"
    ]
    
    all_present = True
    for section in required_sections:
        if section in system_prompt:
            print(f"  ✓ Found: {section}")
        else:
            print(f"  ✗ Missing: {section}")
            all_present = False
    
    if all_present:
        print("✓ PASS: All required sections present")
        return True
    else:
        print("✗ FAIL: Some sections missing")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ToolEvaluator System Prompt Injection Tests")
    print("="*70)
    
    tests = [
        test_system_prompt_injection_with_medical_tools,
        test_no_injection_without_medical_tools,
        test_preserve_existing_system_prompt,
        test_prompt_content_includes_all_sections
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
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
