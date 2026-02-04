#!/usr/bin/env python
"""
Example script demonstrating medical image processing tools with ToolEvaluator.

This script shows how to use:
1. SAM2 (Segment Anything Model 2) for bounding box segmentation
2. BiomedParse for text-based medical image segmentation
3. Zoom-in for region cropping and detailed inspection

Note: This demo requires a running tool server. See the configuration
section below to set up the server URLs.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly to avoid dependency issues
import importlib.util

# Load ToolEvaluator using robust path resolution
tool_eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "tool_evaluator.py")
spec = importlib.util.spec_from_file_location("tool_evaluator", tool_eval_path)
tool_eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool_eval_module)
ToolEvaluator = tool_eval_module.ToolEvaluator

# Load BaseLLM using robust path resolution
base_llm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "base_llm.py")
spec = importlib.util.spec_from_file_location("base_llm", base_llm_path)
base_llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_llm_module)
BaseLLM = base_llm_module.BaseLLM

import numpy as np
from PIL import Image


class SimpleMedicalModel(BaseLLM):
    """A simple mock model for demonstrating medical tool usage."""
    
    def __init__(self):
        super().__init__()
        self.call_count = {}
    
    def process_messages(self, messages):
        return messages
    
    def generate_output(self, messages):
        prompt = messages.get("prompt", "")
        
        # Simulate medical tool call request
        if "segment" in prompt.lower() or "tumor" in prompt.lower():
            # Check if we have tool result already
            has_tool_result = "<tool_result>" in prompt
            
            if not has_tool_result:
                # First call - request SAM2 tool
                return """
Let me segment the tumor region using SAM2.

<tool_call>
SAM2
```json
{
    "index": 1,
    "bbox_2d": [100, 100, 900, 900]
}
```
</tool_call>
"""
            else:
                # Second call - provide final answer after tool result
                return "Based on the segmentation, I can see the tumor is located in the specified region."
        
        elif "zoom" in prompt.lower():
            has_tool_result = "<tool_result>" in prompt
            
            if not has_tool_result:
                return """
Let me zoom into the region of interest.

<tool_call>
Zoom-in
```json
{
    "index": 1,
    "bbox_2d": [200, 200, 800, 800]
}
```
</tool_call>
"""
            else:
                return "The zoomed region shows detailed structures."
        
        elif "biomedparse" in prompt.lower():
            has_tool_result = "<tool_result>" in prompt
            
            if not has_tool_result:
                return """
Let me use BiomedParse to segment the lesion.

<tool_call>
BiomedParse
```json
{
    "index": 1,
    "captions": "lesion"
}
```
</tool_call>
"""
            else:
                return "BiomedParse has identified and segmented the lesion."
        
        else:
            return "This is a standard response without tool calls."
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]


def create_dummy_medical_image():
    """Create a simple dummy medical image for demonstration."""
    # Create a 512x512 grayscale image with a circular "tumor"
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Gray background
    
    # Draw a circular region to represent a tumor
    center_x, center_y = 256, 256
    radius = 80
    
    y, x = np.ogrid[:512, :512]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    img[mask] = [220, 180, 180]  # Slightly darker region
    
    return img


def main():
    """Main demonstration function."""
    
    print("=" * 70)
    print("Medical Image Processing Tools Demo")
    print("=" * 70)
    print()
    
    # 1. Initialize model
    print("1. Initializing medical model...")
    model = SimpleMedicalModel()
    print("   ✓ Model initialized")
    print()
    
    # 2. Create a dummy medical image for demonstration
    print("2. Creating sample medical image...")
    output_dir = "./medical_demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    dummy_image = create_dummy_medical_image()
    image_path = os.path.join(output_dir, "sample_medical_image.png")
    Image.fromarray(dummy_image).save(image_path)
    print(f"   ✓ Sample image created at: {image_path}")
    print()
    
    # 3. Setup ToolEvaluator with medical tools
    print("3. Configuring ToolEvaluator with medical tools...")
    print()
    print("   NOTE: Medical image processing tools (SAM2, BiomedParse) require")
    print("   external tool servers to be running. This demo shows the setup")
    print("   but will not actually connect to servers.")
    print()
    
    # Configure medical tools (without actual server URLs for demo)
    medical_config = {
        "tool_server_url": "http://localhost:6060",  # SAM2 server
        "biomedparse_url": "http://localhost:6061",  # BiomedParse server
        "output_dir": output_dir
    }
    
    # Create evaluator with medical tools
    # Note: Without running servers, medical tools won't execute
    # But we can still demonstrate the setup and basic tools
    try:
        evaluator = ToolEvaluator(
            model=model,
            tools={},  # Start with empty tools
            tool_choice="auto",
            max_tool_calls=5,
            medical_tools_config=medical_config
        )
        
        print(f"   ✓ ToolEvaluator created")
        print(f"   ✓ Registered tools: {', '.join(evaluator.list_tools())}")
        print()
        
        # Show tool details
        print("   Available Medical Tools:")
        print("   • SAM2: Segment Anything Model 2 (bounding box segmentation)")
        print("   • BiomedParse: Text-based medical image segmentation")
        print("   • Zoom-in: Region cropping for detailed inspection")
        print()
        
    except Exception as e:
        print(f"   ⚠ Could not initialize all medical tools: {e}")
        print("   This is expected if tool servers are not running.")
        print()
        
        # Fall back to evaluator without medical tools
        evaluator = ToolEvaluator(
            model=model,
            tools={},
            tool_choice="auto",
            max_tool_calls=5
        )
        print("   ✓ ToolEvaluator created (without external tools)")
        print()
    
    # 4. Demonstrate tool call format
    print("4. Medical Tool Call Format Examples")
    print("-" * 70)
    print()
    print("   SAM2 (Bounding Box Segmentation):")
    print("   <tool_call>")
    print("   SAM2")
    print("   ```json")
    print("   {")
    print('       "index": 1,')
    print('       "bbox_2d": [100, 100, 900, 900]')
    print("   }")
    print("   ```")
    print("   </tool_call>")
    print()
    
    print("   BiomedParse (Text-based Segmentation):")
    print("   <tool_call>")
    print("   BiomedParse")
    print("   ```json")
    print("   {")
    print('       "index": 1,')
    print('       "captions": "lesion"')
    print("   }")
    print("   ```")
    print("   </tool_call>")
    print()
    
    print("   Zoom-in (Region Cropping):")
    print("   <tool_call>")
    print("   Zoom-in")
    print("   ```json")
    print("   {")
    print('       "index": 1,')
    print('       "bbox_2d": [200, 200, 800, 800]')
    print("   }")
    print("   ```")
    print("   </tool_call>")
    print()
    
    # 5. Test parsing without actual execution
    print("5. Testing Tool Call Parsing")
    print("-" * 70)
    
    test_response = """
<tool_call>
SAM2
```json
{
    "index": 1,
    "bbox_2d": [100, 100, 900, 900]
}
```
</tool_call>
"""
    
    tool_call = evaluator._parse_tool_call(test_response)
    if tool_call:
        print(f"   ✓ Successfully parsed tool call:")
        print(f"     Tool: {tool_call.get('name')}")
        print(f"     Arguments: {tool_call.get('arguments')}")
    else:
        print("   ✗ Failed to parse tool call")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ Demo completed successfully!")
    print()
    print("Key Points:")
    print("  • Medical tools extend ToolEvaluator for image processing")
    print("  • SAM2, BiomedParse, and Zoom-in are supported")
    print("  • Tools require external servers for actual execution")
    print("  • Tool call format uses <tool_call> with JSON arguments")
    print("  • Coordinate system: 0-1024 relative coordinates")
    print()
    print("Setup Instructions:")
    print("  1. Start tool servers (SAM2, BiomedParse)")
    print("  2. Configure server URLs in medical_tools_config")
    print("  3. Pass configuration to ToolEvaluator")
    print("  4. Use medical tools in your evaluation workflow")
    print("=" * 70)


if __name__ == "__main__":
    main()
