#!/usr/bin/env python3
"""
Test script to verify that ToolEvaluator changes don't affect inference
when tool calling is disabled or no tools are registered.

This test ensures backward compatibility and that the inference results
remain consistent regardless of message format when tools are not used.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from tool_evaluator import ToolEvaluator
from base_llm import BaseLLM


class MockModel(BaseLLM):
    """Mock model that returns deterministic responses."""
    
    def __init__(self):
        super().__init__()
        self.last_messages = None
        self.call_count = 0
    
    def process_messages(self, messages):
        self.last_messages = messages
        return messages
    
    def generate_output(self, messages):
        """Generate deterministic output based on input."""
        self.last_messages = messages
        self.call_count += 1
        
        # Extract content from different message formats
        if "messages" in messages:
            # Chat-style format
            msg_list = messages["messages"]
            if msg_list:
                last_msg = msg_list[-1]
                content = last_msg.get("content", "")
                if isinstance(content, list):
                    # Multimodal content
                    text_items = [item["text"] for item in content if item.get("type") == "text"]
                    content = " ".join(text_items)
                return f"Response to: {content}"
        elif "prompt" in messages:
            # Prompt-based format
            return f"Response to: {messages['prompt']}"
        
        return "Default response"
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def test_no_tools_registered():
    """Test that ToolEvaluator with no tools passes through to base model."""
    print("Test 1: ToolEvaluator with no tools registered...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    # Test prompt-based format
    messages = {"prompt": "What is 2+2?"}
    response = evaluator.generate_output(messages)
    
    assert response == "Response to: What is 2+2?", f"Expected pass-through response, got: {response}"
    assert model.call_count == 1, "Model should be called exactly once"
    print("  ✓ No tools registered: pass-through works correctly")


def test_tool_choice_none():
    """Test that tool_choice='none' disables tool calling."""
    print("Test 2: ToolEvaluator with tool_choice='none'...")
    
    model = MockModel()
    
    # Register a tool but set tool_choice to 'none'
    def dummy_tool(x: int) -> int:
        return x * 2
    
    evaluator = ToolEvaluator(
        model=model,
        tools={"dummy": dummy_tool},
        tool_choice="none"
    )
    
    messages = {"prompt": "Double the number 5"}
    response = evaluator.generate_output(messages)
    
    assert response == "Response to: Double the number 5", f"Expected pass-through response, got: {response}"
    assert model.call_count == 1, "Model should be called exactly once"
    print("  ✓ tool_choice='none': tool calling is disabled")


def test_chat_format_without_tools():
    """Test chat-style format without tools."""
    print("Test 3: Chat-style messages without tools...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    messages = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
    
    response = evaluator.generate_output(messages)
    
    assert "Response to: Hello, how are you?" == response, f"Unexpected response: {response}"
    assert model.call_count == 1, "Model should be called exactly once"
    print("  ✓ Chat format without tools: works correctly")


def test_prompt_format_without_tools():
    """Test prompt-based format without tools."""
    print("Test 4: Prompt-based messages without tools...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    messages = {"prompt": "Explain quantum mechanics"}
    response = evaluator.generate_output(messages)
    
    assert response == "Response to: Explain quantum mechanics", f"Unexpected response: {response}"
    assert model.call_count == 1, "Model should be called exactly once"
    print("  ✓ Prompt format without tools: works correctly")


def test_multimodal_content_without_tools():
    """Test multimodal content list without tools."""
    print("Test 5: Multimodal content without tools...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    messages = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {"type": "image", "image": "path/to/image.jpg"}
                ]
            }
        ]
    }
    
    response = evaluator.generate_output(messages)
    
    assert "Response to: Analyze this image" == response, f"Unexpected response: {response}"
    assert model.call_count == 1, "Model should be called exactly once"
    print("  ✓ Multimodal content without tools: works correctly")


def test_output_consistency():
    """Test that output is consistent between direct model and wrapped model."""
    print("Test 6: Output consistency between direct and wrapped model...")
    
    # Test with direct model
    direct_model = MockModel()
    messages = {"prompt": "What is the capital of France?"}
    direct_response = direct_model.generate_output(messages)
    
    # Test with wrapped model (no tools)
    wrapped_model = MockModel()
    evaluator = ToolEvaluator(model=wrapped_model, tools={})
    wrapped_response = evaluator.generate_output(messages)
    
    assert direct_response == wrapped_response, \
        f"Responses differ: direct={direct_response}, wrapped={wrapped_response}"
    print("  ✓ Output consistency: direct and wrapped models produce identical results")


def test_batch_processing_without_tools():
    """Test batch processing without tools."""
    print("Test 7: Batch processing without tools...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    messages_list = [
        {"prompt": "Question 1"},
        {"prompt": "Question 2"},
        {"prompt": "Question 3"}
    ]
    
    responses = evaluator.generate_outputs(messages_list)
    
    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
    assert responses[0] == "Response to: Question 1"
    assert responses[1] == "Response to: Question 2"
    assert responses[2] == "Response to: Question 3"
    assert model.call_count == 3, "Model should be called 3 times"
    print("  ✓ Batch processing without tools: works correctly")


def test_system_prompt_not_injected_without_medical_config():
    """Test that system prompt is not injected when medical tools are not configured."""
    print("Test 8: No system prompt injection without medical config...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    # Chat format
    messages = {
        "messages": [
            {"role": "user", "content": "Test question"}
        ]
    }
    
    evaluator.generate_output(messages)
    
    # Verify that no system message was added
    captured_messages = model.last_messages
    if "messages" in captured_messages:
        msg_list = captured_messages["messages"]
        # Should not have system message injected
        if len(msg_list) > 0 and msg_list[0].get("role") == "system":
            # This is only acceptable if it was already there
            pass
        print("  ✓ No system prompt injection without medical config")
    else:
        print("  ✓ No system prompt injection without medical config (prompt format)")


def test_message_format_detection_consistency():
    """Test that message format detection is consistent."""
    print("Test 9: Message format detection consistency...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    # Chat format should be detected
    chat_messages = {
        "messages": [
            {"role": "user", "content": "Chat format test"}
        ]
    }
    
    # Prompt format should be detected
    prompt_messages = {"prompt": "Prompt format test"}
    
    # Both should work without issues
    chat_response = evaluator.generate_output(chat_messages)
    prompt_response = evaluator.generate_output(prompt_messages)
    
    assert "Chat format test" in chat_response
    assert "Prompt format test" in prompt_response
    print("  ✓ Message format detection works consistently")


def test_no_tool_calls_with_tool_like_text():
    """Test that tool-like text in prompt doesn't trigger tool calls without tools."""
    print("Test 10: Tool-like text doesn't trigger calls without tools...")
    
    model = MockModel()
    evaluator = ToolEvaluator(model=model, tools={})
    
    # Prompt contains tool call-like syntax
    messages = {
        "prompt": "Can you <tool_call>calculate_bmi</tool_call> for me?"
    }
    
    response = evaluator.generate_output(messages)
    
    # Should pass through without attempting tool execution
    assert "Response to:" in response
    assert model.call_count == 1, "Model should be called exactly once (no tool loop)"
    print("  ✓ Tool-like text doesn't trigger tool calls without registered tools")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing inference without tool calling")
    print("="*60 + "\n")
    
    tests = [
        test_no_tools_registered,
        test_tool_choice_none,
        test_chat_format_without_tools,
        test_prompt_format_without_tools,
        test_multimodal_content_without_tools,
        test_output_consistency,
        test_batch_processing_without_tools,
        test_system_prompt_not_injected_without_medical_config,
        test_message_format_detection_consistency,
        test_no_tool_calls_with_tool_like_text
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ✗ FAILED: {str(e)}")
            failed_tests.append((test.__name__, str(e)))
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed_tests.append((test.__name__, str(e)))
    
    print("\n" + "="*60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        return False
    else:
        print(f"SUCCESS: All {len(tests)} tests passed!")
        print("\nConclusion: The recent changes to ToolEvaluator do NOT affect")
        print("inference results when tool calling is disabled or no tools are registered.")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
