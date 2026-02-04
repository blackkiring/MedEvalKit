#!/usr/bin/env python
"""
Example script demonstrating ToolEvaluator usage with MedEvalKit.

This script shows how to:
1. Initialize a model with ToolEvaluator
2. Register medical calculation tools
3. Use tools during evaluation
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly to avoid dependency issues in demo
import importlib.util

# Load BaseLLM using robust path resolution
base_llm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "base_llm.py")
spec = importlib.util.spec_from_file_location("base_llm", base_llm_path)
base_llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_llm_module)
BaseLLM = base_llm_module.BaseLLM

# Load ToolEvaluator using robust path resolution
tool_eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "tool_evaluator.py")
spec = importlib.util.spec_from_file_location("tool_evaluator", tool_eval_path)
tool_eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_eval_module)
ToolEvaluator = tool_eval_module.ToolEvaluator


class SimpleModel(BaseLLM):
    """A simple mock model for demonstration purposes."""
    
    def __init__(self):
        super().__init__()
        self.call_count = {}
    
    def process_messages(self, messages):
        return messages
    
    def generate_output(self, messages):
        # For demo, we'll simulate a tool call request on first call,
        # then provide final answer after tool result is available
        prompt = messages.get("prompt", "")
        
        # Track if this is a follow-up call (has tool result)
        has_tool_result = "<tool_result>" in prompt
        
        if "BMI" in prompt or "bmi" in prompt:
            if not has_tool_result:
                # First call - request tool
                return """
I need to calculate the BMI for this patient.

<tool_call>
{
    "name": "calculate_bmi",
    "arguments": {
        "weight_kg": 70,
        "height_m": 1.75
    }
}
</tool_call>
"""
            else:
                # Second call - provide final answer with tool result
                return "Based on the calculation, the patient has a BMI of 22.86, which is in the Normal weight category."
                
        elif "drug dose" in prompt.lower():
            if not has_tool_result:
                return """
Let me calculate the appropriate drug dosage.

<tool_call>
{
    "name": "calculate_drug_dose",
    "arguments": {
        "weight_kg": 70,
        "drug_name": "paracetamol",
        "mg_per_kg": 15
    }
}
</tool_call>
"""
            else:
                return "The calculated dose is 1050mg of paracetamol for this 70kg patient."
        else:
            return "This is a response without tool calls."
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


# Define medical tools
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """
    Calculate BMI and provide interpretation.
    
    Args:
        weight_kg: Patient weight in kilograms
        height_m: Patient height in meters
    
    Returns:
        Dictionary with BMI value and category
    """
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
        "category": category,
        "weight_kg": weight_kg,
        "height_m": height_m
    }


def calculate_drug_dose(weight_kg: float, drug_name: str, mg_per_kg: float) -> dict:
    """
    Calculate drug dosage based on patient weight.
    
    Args:
        weight_kg: Patient weight in kilograms
        drug_name: Name of the drug
        mg_per_kg: Dosage in mg per kg body weight
    
    Returns:
        Dictionary with dosage information
    """
    total_dose_mg = weight_kg * mg_per_kg
    
    return {
        "drug": drug_name,
        "weight_kg": weight_kg,
        "dose_per_kg_mg": mg_per_kg,
        "total_dose_mg": round(total_dose_mg, 2),
        "recommendation": f"Administer {round(total_dose_mg, 2)}mg of {drug_name}"
    }


def get_vital_signs_reference(parameter: str) -> dict:
    """
    Get reference ranges for vital signs.
    
    Args:
        parameter: Vital sign parameter name
    
    Returns:
        Dictionary with reference ranges
    """
    references = {
        "blood_pressure": {
            "normal": "120/80 mmHg",
            "range": "90/60 to 140/90",
            "unit": "mmHg"
        },
        "heart_rate": {
            "normal": "60-100 bpm",
            "range": "40-120",
            "unit": "bpm"
        },
        "temperature": {
            "normal": "37°C (98.6°F)",
            "range": "36.1-37.2°C",
            "unit": "°C"
        },
        "respiratory_rate": {
            "normal": "12-20 breaths/min",
            "range": "10-25",
            "unit": "breaths/min"
        }
    }
    return references.get(parameter, {"error": f"Unknown parameter: {parameter}"})


def main():
    """Main demonstration function."""
    
    print("=" * 70)
    print("ToolEvaluator Demo: Medical Evaluation with Tool Support")
    print("=" * 70)
    print()
    
    # 1. Initialize model
    print("1. Initializing model...")
    model = SimpleModel()
    print("   ✓ Model initialized")
    print()
    
    # 2. Create ToolEvaluator with medical tools
    print("2. Creating ToolEvaluator with medical tools...")
    tools = {
        "calculate_bmi": calculate_bmi,
        "calculate_drug_dose": calculate_drug_dose,
        "get_vital_signs_reference": get_vital_signs_reference
    }
    
    evaluator = ToolEvaluator(
        model=model,
        tools=tools,
        tool_choice="auto",
        max_tool_calls=5
    )
    print(f"   ✓ ToolEvaluator created with {len(tools)} tools")
    print(f"   ✓ Registered tools: {', '.join(evaluator.list_tools())}")
    print()
    
    # 3. Example 1: BMI Calculation
    print("3. Example 1: BMI Calculation with Tool Call")
    print("-" * 70)
    messages1 = {
        "prompt": "A patient weighs 70kg and is 1.75m tall. Calculate their BMI."
    }
    print(f"   Input: {messages1['prompt']}")
    print()
    
    response1 = evaluator.generate_output(messages1)
    print(f"   Output: {response1}")
    print()
    
    history1 = evaluator.get_tool_call_history()
    if history1:
        print("   Tool Calls Made:")
        for call in history1:
            print(f"     - {call['tool_name']}")
            print(f"       Arguments: {call['arguments']}")
    print()
    
    # 4. Example 2: Drug Dosage Calculation
    print("4. Example 2: Drug Dosage Calculation")
    print("-" * 70)
    messages2 = {
        "prompt": "Calculate the drug dose of paracetamol for a 70kg patient."
    }
    print(f"   Input: {messages2['prompt']}")
    print()
    
    response2 = evaluator.generate_output(messages2)
    print(f"   Output: {response2}")
    print()
    
    history2 = evaluator.get_tool_call_history()
    if history2:
        print("   Tool Calls Made:")
        for call in history2:
            print(f"     - {call['tool_name']}")
            print(f"       Arguments: {call['arguments']}")
    print()
    
    # 5. Example 3: Without Tool Calls
    print("5. Example 3: Standard Evaluation (No Tool Calls)")
    print("-" * 70)
    messages3 = {
        "prompt": "What are the symptoms of diabetes?"
    }
    print(f"   Input: {messages3['prompt']}")
    print()
    
    response3 = evaluator.generate_output(messages3)
    print(f"   Output: {response3}")
    print()
    
    history3 = evaluator.get_tool_call_history()
    print(f"   Tool Calls Made: {len(history3)}")
    print()
    
    # 6. Batch Processing Example
    print("6. Example 4: Batch Processing")
    print("-" * 70)
    messages_batch = [
        {"prompt": "Calculate BMI for 70kg, 1.75m"},
        {"prompt": "What is diabetes?"},
        {"prompt": "Calculate drug dose for 65kg patient"}
    ]
    
    print(f"   Processing {len(messages_batch)} messages in batch...")
    responses_batch = evaluator.generate_outputs(messages_batch)
    
    for i, response in enumerate(responses_batch, 1):
        print(f"   Response {i}: {response[:80]}...")
    print()
    
    # 7. Tool Management Example
    print("7. Tool Management")
    print("-" * 70)
    print(f"   Current tools: {evaluator.list_tools()}")
    
    # Add a new tool
    import datetime
    def calculate_age(birth_year: int, current_year: int = None) -> int:
        """Calculate age from birth year (defaults to current year)."""
        if current_year is None:
            current_year = datetime.date.today().year
        return current_year - birth_year
    
    evaluator.register_tool("calculate_age", calculate_age)
    print(f"   After adding 'calculate_age': {evaluator.list_tools()}")
    
    # Remove a tool
    evaluator.unregister_tool("get_vital_signs_reference")
    print(f"   After removing 'get_vital_signs_reference': {evaluator.list_tools()}")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ Demo completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("  • Tool registration and management")
    print("  • Automatic tool call detection and execution")
    print("  • Tool call history tracking")
    print("  • Batch processing with tool support")
    print("  • Seamless integration with existing models")
    print("=" * 70)


if __name__ == "__main__":
    main()
