"""
Unit tests for the ToolEvaluator class.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from utils.tool_evaluator import ToolEvaluator
from models.base_llm import BaseLLM


class MockModel(BaseLLM):
    """Mock model for testing."""
    
    def __init__(self, responses=None):
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
    
    def process_messages(self, messages):
        return messages
    
    def generate_output(self, messages):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Default response"
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def test_tool_evaluator_initialization():
    """Test basic initialization of ToolEvaluator."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    assert evaluator.model == model
    assert evaluator.tools == {}
    assert evaluator.tool_choice == "auto"
    assert evaluator.max_tool_calls == 5
    assert evaluator.tool_call_history == []


def test_tool_evaluator_initialization_with_tools():
    """Test initialization with tools."""
    model = MockModel()
    tools = {
        "add": lambda a, b: a + b,
        "multiply": lambda a, b: a * b
    }
    evaluator = ToolEvaluator(model, tools=tools)
    
    assert len(evaluator.tools) == 2
    assert "add" in evaluator.tools
    assert "multiply" in evaluator.tools


def test_tool_evaluator_invalid_model():
    """Test that ToolEvaluator raises error for invalid model."""
    with pytest.raises(TypeError):
        ToolEvaluator("not a model")


def test_register_tool():
    """Test registering a new tool."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    def custom_tool(x):
        return x * 2
    
    evaluator.register_tool("double", custom_tool)
    assert "double" in evaluator.tools
    assert evaluator.tools["double"](5) == 10


def test_unregister_tool():
    """Test unregistering a tool."""
    model = MockModel()
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools)
    
    evaluator.unregister_tool("add")
    assert "add" not in evaluator.tools


def test_list_tools():
    """Test listing registered tools."""
    model = MockModel()
    tools = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b
    }
    evaluator = ToolEvaluator(model, tools=tools)
    
    tool_list = evaluator.list_tools()
    assert len(tool_list) == 2
    assert "add" in tool_list
    assert "subtract" in tool_list


def test_parse_tool_call_valid():
    """Test parsing valid tool call from response."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    response = """Some text before
<tool_call>
{
    "name": "calculate",
    "arguments": {"expression": "2 + 2"}
}
</tool_call>
Some text after"""
    
    tool_call = evaluator._parse_tool_call(response)
    assert tool_call is not None
    assert tool_call["name"] == "calculate"
    assert tool_call["arguments"]["expression"] == "2 + 2"


def test_parse_tool_call_invalid():
    """Test parsing response without tool call."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    response = "Just a regular response without any tool calls"
    tool_call = evaluator._parse_tool_call(response)
    assert tool_call is None


def test_parse_tool_call_malformed():
    """Test parsing malformed tool call."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    response = "<tool_call>not valid json</tool_call>"
    tool_call = evaluator._parse_tool_call(response)
    assert tool_call is None


def test_execute_tool():
    """Test tool execution."""
    model = MockModel()
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools)
    
    result = evaluator._execute_tool("add", {"a": 5, "b": 3})
    assert result == 8


def test_execute_tool_not_registered():
    """Test executing unregistered tool raises error."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
        evaluator._execute_tool("nonexistent", {})


def test_format_tool_result():
    """Test formatting tool result."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    result_str = evaluator._format_tool_result("calculate", {"sum": 42})
    assert "<tool_result>" in result_str
    assert "</tool_result>" in result_str
    assert "calculate" in result_str
    assert "42" in result_str


def test_generate_output_without_tools():
    """Test generate_output without any tool calls."""
    model = MockModel(responses=["Simple response"])
    evaluator = ToolEvaluator(model)
    
    messages = {"prompt": "What is 2+2?"}
    response = evaluator.generate_output(messages)
    
    assert response == "Simple response"
    assert len(evaluator.tool_call_history) == 0


def test_generate_output_with_tool_call():
    """Test generate_output with a single tool call."""
    # First response requests tool, second response is final
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>',
        "The answer is 4"
    ])
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools)
    
    messages = {"prompt": "What is 2+2?"}
    response = evaluator.generate_output(messages)
    
    assert response == "The answer is 4"
    assert len(evaluator.tool_call_history) == 1
    assert evaluator.tool_call_history[0]["tool_name"] == "add"


def test_generate_output_tool_choice_none():
    """Test that tool_choice='none' disables tool calling."""
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>'
    ])
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools, tool_choice="none")
    
    messages = {"prompt": "What is 2+2?"}
    response = evaluator.generate_output(messages)
    
    # Should return raw response with tool call, not execute it
    assert "<tool_call>" in response
    assert len(evaluator.tool_call_history) == 0


def test_generate_output_max_tool_calls():
    """Test that max_tool_calls limit is respected."""
    # Always return a tool call (infinite loop scenario)
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 1}}\n</tool_call>'
    ] * 10)
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools, max_tool_calls=3)
    
    messages = {"prompt": "Keep calling tools"}
    response = evaluator.generate_output(messages)
    
    # Should stop after max_tool_calls
    assert len(evaluator.tool_call_history) == 3


def test_generate_output_tool_error():
    """Test handling of tool execution errors."""
    model = MockModel(responses=[
        '<tool_call>\n{"name": "divide", "arguments": {"a": 10, "b": 0}}\n</tool_call>'
    ])
    
    def divide(a, b):
        return a / b  # Will raise ZeroDivisionError
    
    tools = {"divide": divide}
    evaluator = ToolEvaluator(model, tools=tools)
    
    messages = {"prompt": "Divide 10 by 0"}
    response = evaluator.generate_output(messages)
    
    # Should return error in response
    assert "<tool_error>" in response
    assert "divide" in response


def test_generate_outputs_batch():
    """Test batch generation with tool calls."""
    model = MockModel(responses=[
        "Response 1",
        "Response 2",
        "Response 3"
    ])
    evaluator = ToolEvaluator(model)
    
    messages_list = [
        {"prompt": "Question 1"},
        {"prompt": "Question 2"},
        {"prompt": "Question 3"}
    ]
    
    responses = evaluator.generate_outputs(messages_list)
    
    assert len(responses) == 3
    assert responses[0] == "Response 1"
    assert responses[1] == "Response 2"
    assert responses[2] == "Response 3"


def test_get_tool_call_history():
    """Test retrieving tool call history."""
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>',
        "Result is 4"
    ])
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools)
    
    messages = {"prompt": "Calculate"}
    evaluator.generate_output(messages)
    
    history = evaluator.get_tool_call_history()
    assert len(history) == 1
    assert history[0]["tool_name"] == "add"
    assert history[0]["call_index"] == 0


def test_reset_tool_call_history():
    """Test resetting tool call history."""
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>',
        "Done"
    ])
    tools = {"add": lambda a, b: a + b}
    evaluator = ToolEvaluator(model, tools=tools)
    
    messages = {"prompt": "Calculate"}
    evaluator.generate_output(messages)
    
    assert len(evaluator.tool_call_history) > 0
    
    evaluator.reset_tool_call_history()
    assert len(evaluator.tool_call_history) == 0


def test_process_messages():
    """Test that process_messages delegates to underlying model."""
    model = MockModel()
    evaluator = ToolEvaluator(model)
    
    messages = {"prompt": "Test"}
    processed = evaluator.process_messages(messages)
    
    # MockModel just returns messages unchanged
    assert processed == messages


def test_multiple_sequential_tool_calls():
    """Test multiple sequential tool calls."""
    model = MockModel(responses=[
        '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>',
        '<tool_call>\n{"name": "multiply", "arguments": {"a": 4, "b": 3}}\n</tool_call>',
        "Final result is 12"
    ])
    tools = {
        "add": lambda a, b: a + b,
        "multiply": lambda a, b: a * b
    }
    evaluator = ToolEvaluator(model, tools=tools)
    
    messages = {"prompt": "Calculate (2+2)*3"}
    response = evaluator.generate_output(messages)
    
    assert response == "Final result is 12"
    assert len(evaluator.tool_call_history) == 2
    assert evaluator.tool_call_history[0]["tool_name"] == "add"
    assert evaluator.tool_call_history[1]["tool_name"] == "multiply"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
