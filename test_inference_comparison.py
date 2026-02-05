#!/usr/bin/env python3
"""
Comparative test to verify that inference results are identical
between direct model calls and ToolEvaluator-wrapped calls when
tool calling is disabled.

This test simulates real-world scenarios to ensure backward compatibility.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from tool_evaluator import ToolEvaluator
from base_llm import BaseLLM


class DeterministicModel(BaseLLM):
    """Model that produces deterministic outputs for comparison testing."""
    
    def __init__(self):
        super().__init__()
        self.responses = {
            "medical_question": "Based on the symptoms described, this appears to be a case of acute appendicitis. The patient should seek immediate medical attention.",
            "image_analysis": "The chest X-ray shows normal cardiac silhouette and clear lung fields. No acute findings.",
            "calculation": "The BMI is 24.5, which falls within the normal range (18.5-24.9).",
            "general": "This is a general response to a medical query."
        }
    
    def _identify_query_type(self, content):
        """Identify the type of query to return appropriate response."""
        content_str = str(content).lower()
        
        if "symptoms" in content_str or "appendicitis" in content_str:
            return "medical_question"
        elif "x-ray" in content_str or "image" in content_str:
            return "image_analysis"
        elif "bmi" in content_str or "calculate" in content_str:
            return "calculation"
        else:
            return "general"
    
    def process_messages(self, messages):
        return messages
    
    def generate_output(self, messages):
        """Generate deterministic output based on input content."""
        content = None
        
        if "messages" in messages:
            # Chat-style format
            msg_list = messages["messages"]
            if msg_list:
                last_msg = msg_list[-1]
                content = last_msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from multimodal content
                    text_items = [item["text"] for item in content if item.get("type") == "text"]
                    content = " ".join(text_items)
        elif "prompt" in messages:
            # Prompt-based format
            content = messages["prompt"]
        
        if content is None:
            content = ""
        
        query_type = self._identify_query_type(content)
        return self.responses[query_type]
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def compare_responses(direct, wrapped, scenario):
    """Compare responses and print results."""
    if direct == wrapped:
        print(f"  ✓ {scenario}: Identical responses")
        return True
    else:
        print(f"  ✗ {scenario}: DIFFERENT responses!")
        print(f"    Direct:  {direct[:100]}...")
        print(f"    Wrapped: {wrapped[:100]}...")
        return False


def test_scenario_1_medical_qa():
    """Test medical Q&A scenario."""
    print("\nScenario 1: Medical Q&A (Prompt format)")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    messages = {
        "prompt": "A 25-year-old patient presents with acute right lower quadrant pain, nausea, and fever. What are the symptoms suggesting?"
    }
    
    direct_response = model_direct.generate_output(messages)
    wrapped_response = evaluator.generate_output(messages)
    
    return compare_responses(direct_response, wrapped_response, "Medical Q&A")


def test_scenario_2_image_analysis():
    """Test image analysis scenario."""
    print("\nScenario 2: Medical Image Analysis (Chat format with multimodal content)")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    messages = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this chest X-ray"},
                    {"type": "image", "image": "path/to/xray.jpg"}
                ]
            }
        ]
    }
    
    direct_response = model_direct.generate_output(messages)
    wrapped_response = evaluator.generate_output(messages)
    
    return compare_responses(direct_response, wrapped_response, "Image Analysis")


def test_scenario_3_calculation():
    """Test calculation scenario."""
    print("\nScenario 3: Medical Calculation (Chat format)")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    messages = {
        "messages": [
            {"role": "user", "content": "Calculate BMI for a patient: height 1.75m, weight 75kg"}
        ]
    }
    
    direct_response = model_direct.generate_output(messages)
    wrapped_response = evaluator.generate_output(messages)
    
    return compare_responses(direct_response, wrapped_response, "Medical Calculation")


def test_scenario_4_batch_processing():
    """Test batch processing scenario."""
    print("\nScenario 4: Batch Processing (Multiple queries)")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    messages_list = [
        {"prompt": "Patient has symptoms of appendicitis"},
        {"prompt": "Review this chest X-ray image"},
        {"prompt": "Calculate BMI"},
        {"prompt": "General medical advice"}
    ]
    
    direct_responses = model_direct.generate_outputs(messages_list)
    wrapped_responses = evaluator.generate_outputs(messages_list)
    
    all_match = True
    for i, (direct, wrapped) in enumerate(zip(direct_responses, wrapped_responses)):
        if direct != wrapped:
            print(f"  ✗ Query {i+1}: DIFFERENT")
            all_match = False
    
    if all_match:
        print(f"  ✓ Batch Processing: All {len(messages_list)} responses identical")
    
    return all_match


def test_scenario_5_mixed_formats():
    """Test mixed message formats."""
    print("\nScenario 5: Mixed Message Formats")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    # Test both formats produce expected results
    prompt_msg = {"prompt": "General medical query"}
    chat_msg = {"messages": [{"role": "user", "content": "General medical query"}]}
    
    prompt_direct = model_direct.generate_output(prompt_msg)
    prompt_wrapped = evaluator.generate_output(prompt_msg)
    
    chat_direct = model_direct.generate_output(chat_msg)
    chat_wrapped = evaluator.generate_output(chat_msg)
    
    result1 = compare_responses(prompt_direct, prompt_wrapped, "Prompt format")
    result2 = compare_responses(chat_direct, chat_wrapped, "Chat format")
    
    return result1 and result2


def test_scenario_6_with_system_messages():
    """Test with system messages in chat format."""
    print("\nScenario 6: Chat with System Message")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    evaluator = ToolEvaluator(model=model_wrapped, tools={})
    
    messages = {
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": "What are the symptoms of appendicitis?"}
        ]
    }
    
    direct_response = model_direct.generate_output(messages)
    wrapped_response = evaluator.generate_output(messages)
    
    return compare_responses(direct_response, wrapped_response, "With System Message")


def test_scenario_7_tool_choice_none():
    """Test with registered tools but tool_choice='none'."""
    print("\nScenario 7: Tools Registered but Disabled (tool_choice='none')")
    
    model_direct = DeterministicModel()
    model_wrapped = DeterministicModel()
    
    # Register tools but disable with tool_choice='none'
    def dummy_tool(x):
        return x * 2
    
    evaluator = ToolEvaluator(
        model=model_wrapped,
        tools={"dummy": dummy_tool},
        tool_choice="none"
    )
    
    messages = {"prompt": "Calculate BMI for patient"}
    
    direct_response = model_direct.generate_output(messages)
    wrapped_response = evaluator.generate_output(messages)
    
    return compare_responses(direct_response, wrapped_response, "Tools Disabled")


def run_comparison_tests():
    """Run all comparison tests."""
    print("="*70)
    print("Inference Comparison Test: Direct vs ToolEvaluator-Wrapped")
    print("="*70)
    print("\nTesting that responses are identical when tool calling is disabled...")
    
    tests = [
        test_scenario_1_medical_qa,
        test_scenario_2_image_analysis,
        test_scenario_3_calculation,
        test_scenario_4_batch_processing,
        test_scenario_5_mixed_formats,
        test_scenario_6_with_system_messages,
        test_scenario_7_tool_choice_none
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append(False)
    
    print("\n" + "="*70)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ SUCCESS: All {total} comparison scenarios passed!")
        print("\n📊 Verification Result:")
        print("   The recent changes to ToolEvaluator do NOT affect inference")
        print("   results when tool calling is disabled or no tools are registered.")
        print("   Direct model calls and ToolEvaluator-wrapped calls produce")
        print("   IDENTICAL outputs across all tested scenarios.")
        return True
    else:
        print(f"❌ FAILED: {total - passed}/{total} scenarios failed")
        return False


if __name__ == "__main__":
    success = run_comparison_tests()
    sys.exit(0 if success else 1)
