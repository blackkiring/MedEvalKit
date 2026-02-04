# ToolEvaluator: Tool-Based Evaluation for MedEvalKit

## Overview

The `ToolEvaluator` class extends MedEvalKit's evaluation framework to support tool-based calls during model inference. This allows models to interact with external tools (e.g., calculators, search engines, databases) during the evaluation process.

## Features

- **Tool Registration**: Dynamically register and unregister tools
- **Automatic Tool Calling**: Models can request tool calls during inference
- **Tool Call History**: Track all tool calls made during evaluation
- **Flexible Configuration**: Control tool selection strategy and limits
- **Backward Compatible**: Works seamlessly with existing MedEvalKit models

## Installation

The ToolEvaluator is included in MedEvalKit. Simply import it:

```python
from utils.tool_evaluator import ToolEvaluator
```

## Basic Usage

### 1. Initialize with a Model

```python
from LLMs import init_llm
from utils.tool_evaluator import ToolEvaluator

# Initialize your model
args = parse_args()
model = init_llm(args)

# Wrap with ToolEvaluator
evaluator = ToolEvaluator(model)
```

### 2. Register Tools

```python
# Define tool functions
def safe_calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    
    SECURITY NOTE: Never use eval() directly in production as it can execute
    arbitrary code. Use a dedicated math parser library instead.
    """
    # For production use, consider libraries like:
    # - py_expression_eval
    # - simpleeval
    # - numexpr
    # For simple cases, ast.literal_eval can parse literals safely
    import ast
    try:
        # Only works for literal expressions
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid expression: {expression}")

def search_database(query: str) -> dict:
    """Search medical database for information."""
    # Your implementation here
    return {"result": "Medical information..."}

# Register tools
evaluator.register_tool("calculate", safe_calculate)
evaluator.register_tool("search", search_database)
```

> **⚠️ Security Warning**: When implementing tools that evaluate expressions or execute code, always use safe alternatives to `eval()`. Direct use of `eval()` can execute arbitrary Python code and poses serious security risks.

### 3. Use in Evaluation

```python
# Generate output with tool support
messages = {
    "prompt": "Calculate the BMI for a patient with weight 70kg and height 1.75m"
}

response = evaluator.generate_output(messages)
print(response)

# Check tool call history
history = evaluator.get_tool_call_history()
for call in history:
    print(f"Called {call['tool_name']} with {call['arguments']}")
```

## Advanced Usage

### Configure Tool Selection

```python
# Disable tool calling
evaluator = ToolEvaluator(model, tool_choice="none")

# Require tool use (not yet implemented)
evaluator = ToolEvaluator(model, tool_choice="required")

# Auto mode (default) - model decides
evaluator = ToolEvaluator(model, tool_choice="auto")
```

### Set Maximum Tool Calls

```python
# Limit to 3 sequential tool calls
evaluator = ToolEvaluator(model, max_tool_calls=3)
```

### Initialize with Tools

```python
tools = {
    "add": lambda a, b: a + b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b != 0 else "Division by zero"
}

evaluator = ToolEvaluator(model, tools=tools)
```

### Dynamic Tool Management

```python
# List registered tools
print(evaluator.list_tools())  # ['add', 'multiply', 'divide']

# Unregister a tool
evaluator.unregister_tool("divide")

# Add a new tool
evaluator.register_tool("square_root", lambda x: x ** 0.5)
```

## Tool Call Format

For a model to request a tool call, it should include the following format in its response:

```
<tool_call>
{
    "name": "tool_name",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2"
    }
}
</tool_call>
```

### Example Model Response

```
To solve this problem, I need to calculate the sum first.

<tool_call>
{
    "name": "calculate",
    "arguments": {
        "expression": "125 + 387"
    }
}
</tool_call>
```

The ToolEvaluator will:
1. Parse the tool call
2. Execute the tool with the provided arguments
3. Format the result
4. Feed it back to the model for a final response

### Example Tool Result Format

After tool execution, the result is formatted and appended to the context:

```
<tool_result>
Tool: calculate
Result: 512
</tool_result>
```

## Complete Example: Medical BMI Calculator

```python
from LLMs import init_llm
from utils.tool_evaluator import ToolEvaluator
from argparse import Namespace

# Define medical tools
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI and provide interpretation."""
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return {
        "bmi": round(bmi, 2),
        "category": category
    }

def get_vital_signs_reference(parameter: str) -> dict:
    """Get reference ranges for vital signs."""
    references = {
        "blood_pressure": {"normal": "120/80 mmHg", "range": "90/60 to 140/90"},
        "heart_rate": {"normal": "60-100 bpm", "range": "40-120"},
        "temperature": {"normal": "37°C (98.6°F)", "range": "36.1-37.2°C"}
    }
    return references.get(parameter, {"error": "Unknown parameter"})

# Initialize model and evaluator
args = Namespace(
    model_name="Qwen2-VL",
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.001,
    repetition_penalty=1.0
)

model = init_llm(args)
evaluator = ToolEvaluator(
    model=model,
    tools={
        "calculate_bmi": calculate_bmi,
        "get_vital_signs_reference": get_vital_signs_reference
    },
    max_tool_calls=5
)

# Evaluate with tools
messages = {
    "prompt": """
    A patient presents with:
    - Weight: 85 kg
    - Height: 1.80 m
    - Blood pressure: 130/85 mmHg
    
    Please calculate their BMI and assess if their blood pressure is within normal range.
    """
}

response = evaluator.generate_output(messages)
print("Model Response:", response)

# Review tool usage
print("\nTool Call History:")
for i, call in enumerate(evaluator.get_tool_call_history(), 1):
    print(f"{i}. {call['tool_name']}({call['arguments']})")
```

## Integration with Existing Evaluation Pipeline

The ToolEvaluator is designed to be a drop-in replacement for standard models:

```python
from benchmarks import prepare_benchmark
from LLMs import init_llm
from utils.tool_evaluator import ToolEvaluator

# Initialize model
model = init_llm(args)

# Wrap with ToolEvaluator (optional - only if you want tool support)
if args.enable_tools:
    model = ToolEvaluator(
        model=model,
        tools=load_tools_config(args.tools_config),
        tool_choice=args.tool_choice,
        max_tool_calls=args.max_tool_calls
    )

# Use in existing evaluation pipeline
for eval_dataset in args.eval_datasets:
    benchmark = prepare_benchmark(model, eval_dataset, dataset_path, output_path)
    benchmark.load_data()
    results = benchmark.eval()
```

## API Reference

### Class: ToolEvaluator

#### Constructor

```python
ToolEvaluator(
    model: BaseLLM,
    tools: Optional[Dict[str, Callable]] = None,
    tool_choice: str = "auto",
    max_tool_calls: int = 5
)
```

**Parameters:**
- `model`: Instance of BaseLLM to wrap
- `tools`: Dictionary mapping tool names to callable functions
- `tool_choice`: Tool selection strategy ("auto", "required", "none")
- `max_tool_calls`: Maximum number of sequential tool calls

#### Methods

**`register_tool(name: str, func: Callable) -> None`**
- Register a new tool function

**`unregister_tool(name: str) -> None`**
- Remove a registered tool

**`list_tools() -> List[str]`**
- Get list of registered tool names

**`generate_output(messages: Dict[str, Any]) -> str`**
- Generate output with tool calling support

**`generate_outputs(messages_list: List[Dict[str, Any]]) -> List[str]`**
- Batch generation with tool support

**`get_tool_call_history() -> List[Dict[str, Any]]`**
- Get history of tool calls from last inference

**`reset_tool_call_history() -> None`**
- Clear the tool call history

**`process_messages(messages: Dict[str, Any]) -> Any`**
- Process messages (delegates to underlying model)

## Error Handling

Tool execution errors are caught and returned in the response:

```
<tool_error>Error executing tool 'divide': division by zero</tool_error>
```

This allows the model to see and potentially handle errors gracefully.

## Security Considerations

### Code Execution
**Never use `eval()` or `exec()` in production tools.** These functions can execute arbitrary Python code and create severe security vulnerabilities:

```python
# ❌ DANGEROUS - Do not use in production
def bad_calculator(expression: str):
    return eval(expression)  # Can execute ANY Python code!

# ✅ SAFE - Use dedicated parsers
def safe_calculator(expression: str):
    from simpleeval import simple_eval
    return simple_eval(expression, names={})
```

### Input Validation
Always validate and sanitize tool inputs:

```python
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    # Validate inputs
    if not isinstance(weight_kg, (int, float)) or weight_kg <= 0:
        raise ValueError("Weight must be a positive number")
    if not isinstance(height_m, (int, float)) or height_m <= 0:
        raise ValueError("Height must be a positive number")
    
    bmi = weight_kg / (height_m ** 2)
    return {"bmi": round(bmi, 2)}
```

### Resource Limits
Set limits to prevent resource exhaustion:

```python
# Limit tool call iterations
evaluator = ToolEvaluator(model, tools=tools, max_tool_calls=10)

# Implement timeouts in tools
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Tool execution timed out")

def safe_tool_with_timeout(arg):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout
    try:
        result = expensive_operation(arg)
    finally:
        signal.alarm(0)  # Cancel alarm
    return result
```

### Data Access Control
Restrict tool access to sensitive data:

```python
def search_database(query: str) -> dict:
    # Sanitize query to prevent SQL injection
    from sqlalchemy import text
    
    # Use parameterized queries
    result = db.execute(
        text("SELECT * FROM medical_records WHERE condition = :query"),
        {"query": query}
    )
    return result.fetchone()
```

## Best Practices

1. **Security First**: NEVER use `eval()` or `exec()` in production tools. Use safe alternatives:
   - For math: Use libraries like `simpleeval`, `py_expression_eval`, or `numexpr`
   - For literals: Use `ast.literal_eval()`
   - For structured data: Use proper JSON/XML parsers
2. **Keep Tools Simple**: Each tool should have a single, well-defined purpose
3. **Handle Errors**: Implement error handling within your tool functions
4. **Document Tools**: Use docstrings to describe tool behavior
5. **Set Reasonable Limits**: Use `max_tool_calls` to prevent infinite loops
6. **Validate Arguments**: Validate and sanitize tool arguments before execution
7. **Test Tools**: Test tools independently before using in evaluation

## Limitations

- Tool calls must follow the exact XML format with JSON payload
- Tools must be pure Python functions (no async support yet)
- No support for streaming tool results
- Tool choice "required" mode not yet implemented

## Future Enhancements

- Support for async tool execution
- Tool schemas for automatic validation
- Parallel tool calling
- Built-in tool library for common medical calculations
- Tool call caching
- Support for tool dependencies

## Contributing

To add new features or report issues with ToolEvaluator, please refer to the main MedEvalKit repository.

## License

Same as MedEvalKit project license.
