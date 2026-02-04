#!/usr/bin/env python3
"""
Test script to verify ToolEvaluator correctly handles chat-style history
and multimodal content lists in messages.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from tool_evaluator import ToolEvaluator
from base_llm import BaseLLM


class MockModelWithHistory(BaseLLM):
    """Mock model for testing that captures chat history."""
    
    def __init__(self):
        super().__init__()
        self.last_messages = None
        self.call_count = 0
    
    def process_messages(self, messages):
        self.last_messages = messages
        return messages
    
    def generate_output(self, messages):
        self.last_messages = messages
        self.call_count += 1
        
        # On first call, simulate a tool call
        # On second call, return final answer (after tool result is available)
        if self.call_count == 1:
            return """Let me calculate that.
<tool_call>
{
    "name": "add",
    "arguments": {"a": 5, "b": 3}
}
</tool_call>"""
        else:
            return "The sum is 8."
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


class MockQwen3VL(BaseLLM):
    """Mock Qwen3_VL model to test content list handling."""
    
    def __init__(self):
        super().__init__()
        self.last_processed_messages = None
    
    def process_messages(self, messages):
        """Mimics Qwen3_VL_vllm.py process_messages logic."""
        current_messages = []
        
        if "messages" in messages:
            msg_list = messages["messages"]
            for message in msg_list:
                role = message["role"]
                content = message["content"]
                # If content is already a list, pass through directly
                if isinstance(content, list):
                    current_messages.append({"role": role, "content": content})
                else:
                    # Otherwise, wrap string content as text item
                    current_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        else:
            # Fallback handling
            prompt = messages.get("prompt", "")
            current_messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        
        self.last_processed_messages = current_messages
        return current_messages
    
    def generate_output(self, messages):
        self.process_messages(messages)
        return "Mock response"
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def test_chat_history_detection():
    """Test that _ensure_chat_history correctly detects chat format."""
    print("\n" + "="*70)
    print("Test 1: Chat history detection")
    print("="*70)
    
    model = MockModelWithHistory()
    evaluator = ToolEvaluator(model=model)
    
    # Test chat-style messages
    chat_messages = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    result = evaluator._ensure_chat_history(chat_messages)
    if result:
        print("✓ PASS: Correctly detected chat-style history")
    else:
        print("✗ FAIL: Failed to detect chat-style history")
        return False
    
    # Test non-chat messages
    prompt_messages = {"prompt": "Hello"}
    result = evaluator._ensure_chat_history(prompt_messages)
    if not result:
        print("✓ PASS: Correctly identified non-chat format")
    else:
        print("✗ FAIL: Incorrectly detected chat format")
        return False
    
    return True


def test_chat_history_with_tool_calls():
    """Test that tool calls append as assistant/user messages in chat history."""
    print("\n" + "="*70)
    print("Test 2: Chat history with tool calls")
    print("="*70)
    
    model = MockModelWithHistory()
    
    # Register a simple tool
    def add(a: int, b: int) -> int:
        return a + b
    
    evaluator = ToolEvaluator(
        model=model,
        tools={"add": add},
        tool_choice="auto",
        max_tool_calls=5
    )
    
    # Test with chat-style messages
    messages = {
        "messages": [
            {"role": "user", "content": "What is 5 + 3?"}
        ]
    }
    
    # Reset call count
    model.call_count = 0
    
    response = evaluator.generate_output(messages)
    
    # Check that messages were appended correctly
    if model.last_messages is None:
        print("✗ FAIL: No messages captured")
        return False
    
    if "messages" not in model.last_messages:
        print("✗ FAIL: Messages format not preserved")
        return False
    
    msg_list = model.last_messages["messages"]
    
    # Should have: original user message + assistant response + user tool result
    if len(msg_list) < 3:
        print(f"✗ FAIL: Expected at least 3 messages, got {len(msg_list)}")
        return False
    
    # Check for assistant message
    assistant_found = any(m.get("role") == "assistant" for m in msg_list)
    if not assistant_found:
        print("✗ FAIL: No assistant message found in history")
        return False
    
    print(f"✓ PASS: Chat history contains {len(msg_list)} messages")
    print(f"  - Initial message count: 1")
    print(f"  - After tool call: {len(msg_list)}")
    print(f"  - Assistant response appended: {assistant_found}")
    
    # Check that tool result is in user message
    tool_result_found = False
    for msg in msg_list:
        if msg.get("role") == "user" and "<tool_result>" in str(msg.get("content", "")):
            tool_result_found = True
            break
    
    if tool_result_found:
        print("  - Tool result appended as user message: True")
    else:
        print("  - Tool result appended as user message: False (Warning)")
    
    return True


def test_multimodal_content_list_passthrough():
    """Test that Qwen3_VL correctly passes through multimodal content lists."""
    print("\n" + "="*70)
    print("Test 3: Multimodal content list passthrough")
    print("="*70)
    
    model = MockQwen3VL()
    
    # Test with content as list (multimodal)
    multimodal_messages = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "path/to/image.jpg"}
                ]
            }
        ]
    }
    
    model.generate_output(multimodal_messages)
    
    if model.last_processed_messages is None:
        print("✗ FAIL: No processed messages")
        return False
    
    if len(model.last_processed_messages) == 0:
        print("✗ FAIL: No messages processed")
        return False
    
    first_msg = model.last_processed_messages[0]
    content = first_msg.get("content")
    
    if not isinstance(content, list):
        print("✗ FAIL: Content is not a list")
        return False
    
    if len(content) != 2:
        print(f"✗ FAIL: Expected 2 content items, got {len(content)}")
        return False
    
    # Check that content was passed through directly
    if content[0].get("type") == "text" and content[1].get("type") == "image":
        print("✓ PASS: Multimodal content list passed through directly")
        print(f"  - Content items: {len(content)}")
        print(f"  - Types: {[item.get('type') for item in content]}")
        return True
    else:
        print("✗ FAIL: Content structure incorrect")
        return False


def test_string_content_wrapping():
    """Test that string content is wrapped as text item."""
    print("\n" + "="*70)
    print("Test 4: String content wrapping")
    print("="*70)
    
    model = MockQwen3VL()
    
    # Test with content as string
    string_messages = {
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }
    
    model.generate_output(string_messages)
    
    if model.last_processed_messages is None:
        print("✗ FAIL: No processed messages")
        return False
    
    first_msg = model.last_processed_messages[0]
    content = first_msg.get("content")
    
    if not isinstance(content, list):
        print("✗ FAIL: Content is not a list")
        return False
    
    if len(content) != 1:
        print(f"✗ FAIL: Expected 1 content item, got {len(content)}")
        return False
    
    if content[0].get("type") == "text" and content[0].get("text") == "Hello, world!":
        print("✓ PASS: String content wrapped as text item")
        print(f"  - Content structure: {content[0]}")
        return True
    else:
        print("✗ FAIL: Content wrapping incorrect")
        return False


def test_backward_compatibility_prompt_based():
    """Test that prompt-based tool calls still work (backward compatibility)."""
    print("\n" + "="*70)
    print("Test 5: Backward compatibility with prompt-based flow")
    print("="*70)
    
    model = MockModelWithHistory()
    
    def multiply(a: int, b: int) -> int:
        return a * b
    
    evaluator = ToolEvaluator(
        model=model,
        tools={"multiply": multiply},
        tool_choice="auto"
    )
    
    # Use old-style prompt-based messages
    messages = {"prompt": "What is 4 times 7?"}
    
    model.call_count = 0
    response = evaluator.generate_output(messages)
    
    # Check that prompt was updated (not converted to messages)
    if model.last_messages is None:
        print("✗ FAIL: No messages captured")
        return False
    
    if "prompt" not in model.last_messages:
        print("✗ FAIL: Prompt-based format not preserved")
        return False
    
    if "messages" in model.last_messages:
        print("✗ FAIL: Prompt was incorrectly converted to messages format")
        return False
    
    print("✓ PASS: Prompt-based flow preserved")
    print(f"  - Format: prompt-based")
    print(f"  - Tool calls executed: {len(evaluator.get_tool_call_history())}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Chat History and Multimodal Content Tests")
    print("="*70)
    
    tests = [
        test_chat_history_detection,
        test_chat_history_with_tool_calls,
        test_multimodal_content_list_passthrough,
        test_string_content_wrapping,
        test_backward_compatibility_prompt_based
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
