"""
ToolEvaluator: A wrapper class for evaluating models with tool-based calls.

This module extends the MedEvalKit framework to support tool-based evaluation,
allowing models to interact with external tools during inference.

SECURITY WARNING: When implementing tools that evaluate expressions or execute
code, use safe alternatives to eval() such as ast.literal_eval() or dedicated
parsers. Direct use of eval() can execute arbitrary code and poses security risks.
"""

import json
from typing import Any, Dict, List, Optional, Callable
from models.base_llm import BaseLLM


class ToolEvaluator:
    """
    A wrapper class that enables tool-based evaluation for medical models.
    
    This class provides an interface for models to make tool calls during evaluation,
    extending the standard evaluation workflow to support more complex interactions.
    
    Args:
        model: The base model instance (must inherit from BaseLLM)
        tools: Optional dictionary of tool functions that can be called
        tool_choice: Strategy for tool selection ("auto", "required", "none")
        max_tool_calls: Maximum number of tool calls allowed per inference
    
    Example:
        >>> model = init_llm(args)
        >>> # SECURITY NOTE: Use safe math parsers instead of eval() in production
        >>> # Example shown for demonstration only
        >>> def safe_calculate(expression: str) -> float:
        ...     # Use a proper math parser library like py_expression_eval
        ...     # or ast.literal_eval for safe evaluation
        ...     import ast
        ...     return ast.literal_eval(expression)
        >>> tools = {
        ...     "calculate": safe_calculate,
        ...     "search": lambda q: search_database(q)
        ... }
        >>> tool_evaluator = ToolEvaluator(model, tools=tools)
        >>> result = tool_evaluator.generate_output(messages)
    """
    
    def __init__(
        self,
        model: BaseLLM,
        tools: Optional[Dict[str, Callable]] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5
    ):
        """
        Initialize the ToolEvaluator with a base model and optional tools.
        
        Args:
            model: The underlying model to wrap
            tools: Dictionary mapping tool names to callable functions
            tool_choice: How tools should be selected ("auto", "required", "none")
            max_tool_calls: Maximum number of sequential tool calls
        """
        if not isinstance(model, BaseLLM):
            # Check if model has the required methods instead of strict type checking
            required_methods = ['process_messages', 'generate_output', 'generate_outputs']
            if not all(hasattr(model, method) for method in required_methods):
                raise TypeError("model must be an instance of BaseLLM or implement its interface")
        
        self.model = model
        self.tools = tools or {}
        self.tool_choice = tool_choice
        self.max_tool_calls = max_tool_calls
        self.tool_call_history: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, func: Callable) -> None:
        """
        Register a new tool function.
        
        Args:
            name: Name of the tool
            func: Callable function to register
        """
        self.tools[name] = func
    
    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool function.
        
        Args:
            name: Name of the tool to remove
        """
        if name in self.tools:
            del self.tools[name]
    
    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from model response.
        
        Expected format in response:
        <tool_call>
        {
            "name": "tool_name",
            "arguments": {...}
        }
        </tool_call>
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with tool name and arguments, or None if no tool call found
        """
        try:
            # Look for tool call markers
            if "<tool_call>" in response and "</tool_call>" in response:
                start = response.index("<tool_call>") + len("<tool_call>")
                end = response.index("</tool_call>")
                tool_call_str = response[start:end].strip()
                tool_call = json.loads(tool_call_str)
                return tool_call
        except (ValueError, json.JSONDecodeError):
            pass
        return None
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Result from tool execution
            
        Raises:
            ValueError: If tool is not registered
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        tool_func = self.tools[tool_name]
        return tool_func(**arguments)
    
    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format tool execution result for model input.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result from tool execution
            
        Returns:
            Formatted string to append to model context
        """
        return f"\n<tool_result>\nTool: {tool_name}\nResult: {json.dumps(result, default=str)}\n</tool_result>\n"
    
    def process_messages(self, messages: Dict[str, Any]) -> Any:
        """
        Process messages using the underlying model.
        
        Args:
            messages: Input messages dictionary
            
        Returns:
            Processed messages suitable for model input
        """
        return self.model.process_messages(messages)
    
    def generate_output(self, messages: Dict[str, Any]) -> str:
        """
        Generate output with tool calling support.
        
        This method extends the base model's generate_output to support
        tool calls. If the model requests a tool call, it executes the tool
        and feeds the result back to the model.
        
        Args:
            messages: Input messages dictionary
            
        Returns:
            Final model response after any tool calls
        """
        self.tool_call_history = []
        current_messages = messages.copy()
        
        # Tool calling is disabled if tool_choice is "none" or no tools registered
        if self.tool_choice == "none" or not self.tools:
            return self.model.generate_output(current_messages)
        
        # Iterative tool calling loop
        for call_idx in range(self.max_tool_calls):
            response = self.model.generate_output(current_messages)
            
            # Check if response contains a tool call
            tool_call = self._parse_tool_call(response)
            
            if tool_call is None:
                # No tool call found, return final response
                return response
            
            # Execute the tool
            try:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                # Record tool call
                self.tool_call_history.append({
                    "call_index": call_idx,
                    "tool_name": tool_name,
                    "arguments": arguments
                })
                
                # Execute tool
                result = self._execute_tool(tool_name, arguments)
                
                # Format result and append to context
                tool_result_str = self._format_tool_result(tool_name, result)
                
                # Update messages with tool result
                if "prompt" in current_messages:
                    current_messages["prompt"] += tool_result_str
                else:
                    # Handle different message formats
                    current_messages = messages.copy()
                    if "prompt" not in current_messages:
                        current_messages["prompt"] = ""
                    current_messages["prompt"] += f"\nPrevious response: {response}\n"
                    current_messages["prompt"] += tool_result_str
                
            except Exception as e:
                # If tool execution fails, return error in response
                error_msg = f"\n<tool_error>Error executing tool '{tool_name}': {str(e)}</tool_error>\n"
                return response + error_msg
        
        # Max tool calls reached, return last response
        return response
    
    def generate_outputs(self, messages_list: List[Dict[str, Any]]) -> List[str]:
        """
        Generate outputs for a batch of messages with tool calling support.
        
        Args:
            messages_list: List of message dictionaries
            
        Returns:
            List of response strings
        """
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results
    
    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of tool calls from the last inference.
        
        Returns:
            List of tool call records
        """
        return self.tool_call_history
    
    def reset_tool_call_history(self) -> None:
        """
        Clear the tool call history.
        """
        self.tool_call_history = []
