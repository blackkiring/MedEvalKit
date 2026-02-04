#!/usr/bin/env python3
"""
Demo script to show chat history and multimodal content working end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from tool_evaluator import ToolEvaluator
from base_llm import BaseLLM


class SimpleModel(BaseLLM):
    """Simple model that demonstrates chat history usage."""
    
    def __init__(self):
        super().__init__()
        self.generation_count = 0
    
    def process_messages(self, messages):
        """Process messages (mimics Qwen3_VL behavior)."""
        if "messages" in messages:
            print(f"\n[Model] Processing {len(messages['messages'])} messages in chat history")
            for i, msg in enumerate(messages["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content_preview = f"[multimodal: {len(content)} items]"
                else:
                    content_preview = content[:50] + "..." if len(str(content)) > 50 else content
                print(f"  {i+1}. {role}: {content_preview}")
        else:
            print(f"\n[Model] Processing prompt-based message")
        return messages
    
    def generate_output(self, messages):
        self.process_messages(messages)
        self.generation_count += 1
        
        # Simulate tool call on first generation
        if self.generation_count == 1:
            return """Let me use a tool to help answer that.
<tool_call>
{
    "name": "get_info",
    "arguments": {"query": "medical guidelines"}
}
</tool_call>"""
        else:
            return "Based on the tool result, here is my final answer: The guidelines recommend X, Y, and Z."
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def demo_chat_history_with_tools():
    """Demonstrate chat history with tool calls."""
    print("\n" + "="*70)
    print("DEMO: Chat History with Tool Calls")
    print("="*70)
    
    model = SimpleModel()
    
    # Register a tool
    def get_info(query: str) -> dict:
        print(f"\n[Tool Execution] get_info(query='{query}')")
        return {"result": f"Information about {query}: Lorem ipsum..."}
    
    evaluator = ToolEvaluator(
        model=model,
        tools={"get_info": get_info},
        max_tool_calls=3
    )
    
    # Use chat-style messages
    messages = {
        "messages": [
            {"role": "user", "content": "What are the medical guidelines for treatment?"}
        ]
    }
    
    print("\nInitial message format: chat-style history")
    print(f"Messages: {messages['messages']}")
    
    response = evaluator.generate_output(messages)
    
    print(f"\n[Final Response]\n{response}")
    print(f"\nTool call history: {len(evaluator.get_tool_call_history())} calls")


def demo_multimodal_content():
    """Demonstrate multimodal content passthrough."""
    print("\n" + "="*70)
    print("DEMO: Multimodal Content List Handling")
    print("="*70)
    
    # Simulate Qwen3_VL processing
    class Qwen3VLMock(BaseLLM):
        def process_messages(self, messages):
            current_messages = []
            
            if "messages" in messages:
                msg_list = messages["messages"]
                for message in msg_list:
                    role = message["role"]
                    content = message["content"]
                    # If content is already a list, pass through directly
                    if isinstance(content, list):
                        current_messages.append({"role": role, "content": content})
                        print(f"[Qwen3_VL] Passed through multimodal content: {len(content)} items")
                    else:
                        # Otherwise, wrap string content
                        current_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                        print(f"[Qwen3_VL] Wrapped string content")
            
            return current_messages
        
        def generate_output(self, messages):
            processed = self.process_messages(messages)
            return "Image analysis complete."
        
        def generate_outputs(self, messages_list):
            return [self.generate_output(m) for m in messages_list]
    
    model = Qwen3VLMock()
    
    # Test with multimodal content
    messages = {
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
    
    print("\nInput message with multimodal content:")
    print(f"  Role: {messages['messages'][0]['role']}")
    print(f"  Content: {messages['messages'][0]['content']}")
    
    response = model.generate_output(messages)
    print(f"\nResponse: {response}")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with prompt-based flow."""
    print("\n" + "="*70)
    print("DEMO: Backward Compatibility (Prompt-based)")
    print("="*70)
    
    model = SimpleModel()
    model.generation_count = 0  # Reset
    
    def calculate(x: int, y: int) -> int:
        print(f"\n[Tool Execution] calculate(x={x}, y={y})")
        return x + y
    
    evaluator = ToolEvaluator(
        model=model,
        tools={"calculate": calculate}
    )
    
    # Use old-style prompt-based messages
    messages = {"prompt": "What is 10 plus 5?"}
    
    print("\nInitial message format: prompt-based (legacy)")
    print(f"Prompt: {messages['prompt']}")
    
    response = evaluator.generate_output(messages)
    
    print(f"\n[Final Response]\n{response}")
    print(f"\nTool calls: {len(evaluator.get_tool_call_history())}")


def main():
    print("\n" + "="*70)
    print("Chat History & Multimodal Content - End-to-End Demo")
    print("="*70)
    
    demo_chat_history_with_tools()
    demo_multimodal_content()
    demo_backward_compatibility()
    
    print("\n" + "="*70)
    print("✅ All demos completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
