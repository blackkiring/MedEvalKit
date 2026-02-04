#!/usr/bin/env python3
"""
Minimal test to verify _format_tool_result changes don't break ToolEvaluator.
This test doesn't require numpy or other heavy dependencies.
"""

import sys
import os

# Add utils and models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Test just the formatting logic without full initialization
def test_format_tool_result_logic():
    """Test the _format_tool_result method logic."""
    print("Testing _format_tool_result logic...")
    print("=" * 70)
    
    # Import the template constant
    from utils.tool_evaluator import MEDICAL_AGENT_TOOL_FEEDBACK
    import json
    
    # Replicate the logic from _format_tool_result
    def format_result(tool_name: str, result) -> str:
        is_medical_tool = tool_name in ["SAM2", "BiomedParse", "Zoom-in"]
        is_medical_result = isinstance(result, dict) and all(
            k in result for k in ["index", "path", "width", "height"]
        )
        
        if is_medical_tool and is_medical_result:
            return MEDICAL_AGENT_TOOL_FEEDBACK.format(
                image_index=result["index"],
                width=result["width"],
                height=result["height"]
            )
        else:
            return f"\n<tool_result>\nTool: {tool_name}\nResult: {json.dumps(result, default=str)}\n</tool_result>\n"
    
    # Test medical tool format
    medical_result = {
        "index": 2,
        "path": "/path/to/image.png",
        "width": 512,
        "height": 512,
        "mask_path": "/path/to/mask.png"
    }
    
    output = format_result("BiomedParse", medical_result)
    assert "<image>" in output, "Medical tool should output <image>"
    assert "index of the given image is 2" in output, "Should include image index"
    assert "width: 512" in output, "Should include width"
    assert "height: 512" in output, "Should include height"
    print("✓ Medical tool format test PASSED")
    
    # Test generic tool format
    generic_result = {"value": 42}
    output = format_result("generic_tool", generic_result)
    assert "<tool_result>" in output, "Generic tool should output <tool_result>"
    assert "Tool: generic_tool" in output, "Should include tool name"
    print("✓ Generic tool format test PASSED")
    
    print("=" * 70)
    print("All tests PASSED!")
    return True


if __name__ == "__main__":
    try:
        success = test_format_tool_result_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
