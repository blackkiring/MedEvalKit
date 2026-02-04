"""
ToolEvaluator: A wrapper class for evaluating models with tool-based calls.

This module extends the MedEvalKit framework to support tool-based evaluation,
allowing models to interact with external tools during inference.

SECURITY WARNING: When implementing tools that evaluate expressions or execute
code, use safe alternatives to eval() such as ast.literal_eval() or dedicated
parsers. Direct use of eval() can execute arbitrary code and poses security risks.
"""

import json
import os
import re
import sys
import uuid
import io
import shutil
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Callable, Tuple
from models.base_llm import BaseLLM

# Import medical system prompts
try:
    # Try relative import from examples directory
    examples_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
    if examples_path not in sys.path:
        sys.path.insert(0, examples_path)
    from medical_agent import (
        MEDICAL_AGENT_TOOL_DESCRIPTION,
        MEDICAL_AGENT_REQUIRED_FORMAT,
        MEDICAL_AGENT_REASONING_TIPS,
        MEDICAL_AGENT_SYSTEM_PROMPT,
        MEDICAL_AGENT_USER_PROMPT,
        MEDICAL_AGENT_TOOL_FEEDBACK
    )
except ImportError:
    # Fallback: Define prompts inline if import fails
    MEDICAL_AGENT_TOOL_DESCRIPTION = """### Available tools:
You can use the following three tools to process the images. After each tool usage, you must wait for and analyze the visualization feedback before proceeding.

1. **Zoom-in**
- Purpose: Zoom in on a specific region of an image by cropping it to a bounding box for detailed inspection. If a mask is provided, the zoomed image will highlight the mask's contour.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "bbox_2d": [x1, y1, x2, y2]
    }
]
```
- Output: Generates zoomed areas for visual inspection of the i-th image

2. **BiomedParse**
- Purpose: Detect and segment a specified object type in the image (e.g. lesion, tumor) using text descriptions for the targets.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "captions": "target_description"
    }
]
```
- Output: Generates segmentation masks for target objects of the i-th image

3. **SAM2**
- Purpose: Detect and Segment an object in the image given a bounding box.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "bbox_2d": [x1, y1, x2, y2]
    }
]
```
- Output: Generates segmentation masks for target objects of the i-th image
"""

    MEDICAL_AGENT_REQUIRED_FORMAT = """### Required Output Format:
For each reasoning step, you must structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: [Zoom-in/BiomedParse/SAM2]
```json
[JSON format coordinates or descriptions]
```

After your reasoning and iteratively refine your solution through tool invocation feedback, you should arrive at a final answer and structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: Answer
<answer> [Your final answer] </answer>
"""

    MEDICAL_AGENT_REASONING_TIPS = """### Please NOTE the following reasoning techniques:
1. Initial Analysis
   - Break down the complex problem
   - Plan your approach

2. Iterative Reasoning for Each Step
   - Choose appropriate tool
   - Provide absolute coordinates in JSON format (The top-left corner of the image is (0, 0) and the bottom-right corner is (512, 512))
   - Observe the tool invocation output
   - Reflect on the results returned by the tool:
     * Does the results of the segmentation or zooming reasonable?
     * Does it align with your reasoning?
     * What adjustments are needed?
   - Backtrack and Adjust:
     * If errors found, backtrack to previous step to modify actions or decisions as needed.
"""

    MEDICAL_AGENT_SYSTEM_PROMPT = f"""### Guidance:
You are a helpful assistant specialized in medical image analysis. You have access to several tools that help you segment and examine medical images (e.g. highlighting lesions or tumors) to answer questions.
Your task is to carefully analyze the image and question, use the tools step-by-step, and provide a well-reasoned final answer through tool invocation feedback.

{MEDICAL_AGENT_TOOL_DESCRIPTION}

{MEDICAL_AGENT_REQUIRED_FORMAT}

{MEDICAL_AGENT_REASONING_TIPS}
"""

    MEDICAL_AGENT_USER_PROMPT = """<image>
### Question:
{question}
Options:
{options}
The index of the given image is {image_index} (width: {width}, height: {height}).
Begin your reasoning. After each tool use, critically evaluate the image returned by the tool and adjust tool decisions if needed:
"""

    MEDICAL_AGENT_TOOL_FEEDBACK = """<image>
The index of the given image is {image_index} (width: {width}, height: {height}). Continue your reasoning. After each tool use, critically evaluate the image returned by the tool and adjust tool decisions if needed:
"""


class MedicalToolClient:
    """Tool server client for medical image processing tools."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the medical tool client.
        
        Args:
            base_url: Base URL of the tool server
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    def run(self, endpoint: str, payload: dict, expect_binary: bool = False) -> Any:
        """
        Call tool endpoint.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload dictionary
            expect_binary: Whether to expect binary response
            
        Returns:
            Response data (JSON or binary)
            
        Raises:
            RuntimeError: If tool request fails
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()  # Raise HTTP errors
            if expect_binary:
                return response.content
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Tool request failed: {str(e)}") from e


class ToolEvaluator:
    """
    A wrapper class that enables tool-based evaluation for medical models.
    
    This class provides an interface for models to make tool calls during evaluation,
    extending the standard evaluation workflow to support more complex interactions.
    Supports both simple function tools and advanced medical image processing tools.
    
    When medical tools are configured, ToolEvaluator automatically injects comprehensive
    system prompts that include:
    - Tool descriptions (SAM2, BiomedParse, Zoom-in)
    - Required output format specifications
    - Reasoning tips for iterative tool usage
    
    Args:
        model: The base model instance (must inherit from BaseLLM)
        tools: Optional dictionary of tool functions that can be called
        tool_choice: Strategy for tool selection ("auto", "required", "none")
        max_tool_calls: Maximum number of tool calls allowed per inference
        medical_tools_config: Optional configuration for medical image processing tools
            - tool_server_url: URL of the main tool server (for SAM2)
            - biomedparse_url: URL of the BiomedParse server
            - output_dir: Directory for saving processed images
            
    Note:
        The medical system prompt is automatically injected into messages when:
        1. medical_tools_config is provided
        2. No "system" key exists in the input messages
        This ensures models receive proper guidance without requiring manual prompt setup.
    
    Example:
        >>> model = init_llm(args)
        >>> # Simple tools
        >>> def safe_calculate(expression: str) -> float:
        ...     import ast
        ...     return ast.literal_eval(expression)
        >>> tools = {"calculate": safe_calculate}
        >>> 
        >>> # With medical tools (system prompt automatically injected)
        >>> medical_config = {
        ...     "tool_server_url": "http://localhost:6060",
        ...     "biomedparse_url": "http://localhost:6061",
        ...     "output_dir": "./medical_outputs"
        ... }
        >>> tool_evaluator = ToolEvaluator(
        ...     model, 
        ...     tools=tools,
        ...     medical_tools_config=medical_config
        ... )
        >>> # System prompt with tool descriptions is automatically added
        >>> result = tool_evaluator.generate_output(messages)
    """
    
    # Constants for medical image processing
    MIN_BBOX_SIZE = 10  # Minimum bounding box dimensions in pixels
    
    def __init__(
        self,
        model: BaseLLM,
        tools: Optional[Dict[str, Callable]] = None,
        tool_choice: str = "auto",
        max_tool_calls: int = 5,
        medical_tools_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ToolEvaluator with a base model and optional tools.
        
        Args:
            model: The underlying model to wrap
            tools: Dictionary mapping tool names to callable functions
            tool_choice: How tools should be selected ("auto", "required", "none")
            max_tool_calls: Maximum number of sequential tool calls
            medical_tools_config: Configuration for medical image processing tools
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
        
        # Medical tools configuration
        self.medical_tools_config = medical_tools_config or {}
        self._setup_medical_tools()
    
    def _setup_medical_tools(self) -> None:
        """
        Setup medical image processing tools if configuration is provided.
        
        Medical tools include:
        - SAM2: Segment Anything Model 2 for bounding box segmentation
        - BiomedParse: Text-based medical image segmentation
        - Zoom-in: Region cropping for detailed inspection
        """
        if not self.medical_tools_config:
            return
        
        # Setup output directories for medical tools
        output_dir = Path(self.medical_tools_config.get("output_dir", "./medical_tool_outputs"))
        self.mask_dir = output_dir / "masks"
        self.overlay_dir = output_dir / "overlays"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tool clients
        tool_server_url = self.medical_tools_config.get("tool_server_url", "")
        biomedparse_url = self.medical_tools_config.get("biomedparse_url", "")
        
        self._tool_clients = {}
        self.tool_endpoints = {}
        
        if tool_server_url:
            self._tool_clients["SAM2"] = MedicalToolClient(tool_server_url)
            self.tool_endpoints["SAM2"] = "/segment"
        
        if biomedparse_url:
            self._tool_clients["BiomedParse"] = MedicalToolClient(biomedparse_url)
            self.tool_endpoints["BiomedParse"] = "/biomedparse"
        
        # Register medical tools
        if "SAM2" in self._tool_clients:
            self.register_tool("SAM2", self._handle_sam2)
        if "BiomedParse" in self._tool_clients:
            self.register_tool("BiomedParse", self._handle_biomedparse)
        self.register_tool("Zoom-in", self._handle_zoom_in)
        
        # Image cache for medical tools
        self._image_cache = []
        self._images = []
    
    def _register_image(self, path: str, mask_path: Optional[str] = None, arr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Register image to cache and record metadata.
        
        Args:
            path: Path to the image file
            mask_path: Optional path to associated mask file
            arr: Optional numpy array of the image
            
        Returns:
            Dictionary with image metadata
        """
        if arr is None:
            arr = self._load_image_array(path)
        
        width, height = self._read_image_size(path)
        self._image_cache.append(arr)
        
        meta = {
            "index": len(self._images) + 1,
            "path": path,
            "width": width,
            "height": height,
            "mask_path": mask_path
        }
        self._images.append(meta)
        return meta
    
    def _read_image_size(self, img_path: str) -> Tuple[int, int]:
        """Read image dimensions."""
        with Image.open(img_path) as img:
            return img.width, img.height
    
    def _load_image_array(self, img_path: str) -> np.ndarray:
        """Load image as numpy array in RGB format."""
        img = Image.open(img_path).convert("RGB")
        return np.array(img)
    
    def _get_image_by_index(self, idx: int) -> Dict[str, Any]:
        """Get image metadata by index."""
        if idx < 1 or idx > len(self._images):
            raise IndexError(f"Image index {idx} out of range (1-{len(self._images)})")
        return self._images[idx - 1]
    
    def _rescale_and_clamp_bbox(self, bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Convert relative bbox coordinates (0-1024) to pixel coordinates and clamp to image bounds.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in 0-1024 coordinates
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates
        """
        x1, y1, x2, y2 = bbox
        # Convert relative coordinates to pixels
        x1 = (x1 / 1024.0) * width
        y1 = (y1 / 1024.0) * height
        x2 = (x2 / 1024.0) * width
        y2 = (y2 / 1024.0) * height
        
        # Round and clamp to bounds
        x1 = max(0, min(int(round(x1)), width))
        y1 = max(0, min(int(round(y1)), height))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))
        
        # Fix invalid bounding boxes using minimum size
        if x2 <= x1:
            x2 = min(x1 + self.MIN_BBOX_SIZE, width)
        if y2 <= y1:
            y2 = min(y1 + self.MIN_BBOX_SIZE, height)
        
        return x1, y1, x2, y2
    
    def _create_overlay_array(self, img_np: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Create mask overlay visualization (red mask on original image).
        
        Args:
            img_np: Original image as numpy array
            mask: Mask array (binary or probability)
            alpha: Overlay transparency
            
        Returns:
            Overlay image as numpy array
        """
        h, w = img_np.shape[:2]
        
        # Resize mask if needed
        if mask.shape != (h, w):
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0
        
        # Create red overlay for mask region
        mask_bin = mask > 0.5
        overlay = img_np.copy().astype(np.float32)
        overlay[mask_bin] = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        
        # Blend with original image
        vis = (img_np.astype(np.float32) * (1.0 - alpha) + overlay * alpha)
        return vis.clip(0, 255).astype(np.uint8)
    
    def _load_mask_array_from_path(self, path: str, hw: Tuple[int, int]) -> np.ndarray:
        """Load mask and resize to specified dimensions."""
        with Image.open(path) as m:
            m = m.convert("L").resize((hw[1], hw[0]), Image.NEAREST)
            return np.array(m).astype(np.float32) / 255.0
    
    def _call_segmenter(self, tool: str, image_path: str, extra: Dict[str, Any], expect_binary: bool = False) -> Any:
        """
        Call medical segmentation tool server.
        
        Args:
            tool: Tool name (SAM2 or BiomedParse)
            image_path: Path to image file
            extra: Additional parameters
            expect_binary: Whether to expect binary response
            
        Returns:
            Response data (binary or path)
        """
        endpoint = self.tool_endpoints.get(tool)
        if not endpoint:
            raise ValueError(f"No endpoint found for tool: {tool}")
        
        client = self._tool_clients.get(tool)
        payload = {"image": image_path, "image_path": image_path}
        payload.update(extra)
        
        response = client.run(endpoint, payload, expect_binary=expect_binary)
        
        if isinstance(response, (bytes, bytearray)):
            return response
        return response.get("mask_path") or response.get("image_path")
    
    def _handle_sam2(self, **kwargs) -> Dict[str, Any]:
        """
        SAM2 tool handler: Segment image region using bounding box.
        
        Args:
            index: Image index (1-based)
            bbox_2d: Bounding box [x1, y1, x2, y2] in 0-1024 coordinates
            
        Returns:
            Dictionary with new image metadata
        """
        idx = int(kwargs.get("index", 1))
        bbox = kwargs.get("bbox_2d")
        
        if not bbox or len(bbox) != 4:
            raise ValueError("SAM2 requires valid bbox_2d: [x1, y1, x2, y2]")
        
        # Get base image from cache
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img_path = base_image_meta["path"]
        h, w = base_arr.shape[:2]
        
        # Rescale and clamp bbox coordinates
        rescaled_bbox = list(self._rescale_and_clamp_bbox(bbox, w, h))
        extra = {"bbox": rescaled_bbox, "bbox_2d": rescaled_bbox}
        
        # Add optional parameters
        for k in ["clicklist", "labels", "multimask_output", "return_logits"]:
            if k in kwargs:
                extra[k] = kwargs[k]
        
        # Call tool server to get mask
        npz_bytes = self._call_segmenter("SAM2", img_path, extra, expect_binary=True)
        
        # Parse mask (take the one with highest confidence)
        with np.load(io.BytesIO(npz_bytes)) as data:
            masks = data.get("masks")
            scores = data.get("scores")
        
        if masks is None:
            raise RuntimeError("SAM2 returned no masks")
        
        if masks.ndim == 3:
            best_idx = int(np.argmax(scores)) if scores is not None else 0
            best_mask = masks[best_idx].astype(np.float32)
        else:
            best_mask = masks.astype(np.float32)
        
        # Save mask
        mask_path = str(self.mask_dir / f"sam2_mask_{uuid.uuid4().hex}.png")
        Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255).save(mask_path)
        
        # Create overlay visualization
        overlay_arr = self._create_overlay_array(base_arr, best_mask, alpha=0.5)
        overlay_path = str(self.overlay_dir / f"sam2_overlay_{uuid.uuid4().hex}.png")
        Image.fromarray(overlay_arr).save(overlay_path)
        
        # Register new image
        return self._register_image(
            path=str(Path(overlay_path).resolve()),
            mask_path=str(Path(mask_path).resolve()),
            arr=overlay_arr
        )
    
    def _handle_biomedparse(self, **kwargs) -> Dict[str, Any]:
        """
        BiomedParse tool handler: Segment image using text description.
        
        Args:
            index: Image index (1-based)
            captions: Text description of target to segment
            
        Returns:
            Dictionary with new image metadata
        """
        idx = int(kwargs.get("index", 1))
        captions = kwargs.get("captions", "")
        
        if not captions or not isinstance(captions, str):
            raise ValueError("BiomedParse requires non-empty captions string")
        
        # Get base image from cache
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img_path = base_image_meta["path"]
        h, w = base_arr.shape[:2]
        
        # Call tool server to get mask path
        tool_mask_path = self._call_segmenter("BiomedParse", img_path, {"captions": captions})
        
        if not tool_mask_path or not os.path.exists(tool_mask_path):
            raise RuntimeError("BiomedParse did not return a valid mask path")
        
        # Copy mask to output directory
        mask_dst_path = str(self.mask_dir / f"biomedparse_mask_{uuid.uuid4().hex}.png")
        shutil.copy2(tool_mask_path, mask_dst_path)
        
        # Load mask and create overlay
        mask_arr = self._load_mask_array_from_path(mask_dst_path, (h, w))
        overlay_arr = self._create_overlay_array(base_arr, mask_arr, alpha=0.5)
        overlay_path = str(self.overlay_dir / f"biomedparse_overlay_{uuid.uuid4().hex}.png")
        Image.fromarray(overlay_arr).save(overlay_path)
        
        # Register new image
        return self._register_image(
            path=str(Path(overlay_path).resolve()),
            mask_path=str(Path(mask_dst_path).resolve()),
            arr=overlay_arr
        )
    
    def _handle_zoom_in(self, **kwargs) -> Dict[str, Any]:
        """
        Zoom-in tool handler: Crop image region using bounding box.
        
        Args:
            index: Image index (1-based)
            bbox_2d: Bounding box [x1, y1, x2, y2] in 0-1024 coordinates
            
        Returns:
            Dictionary with new image metadata
        """
        idx = int(kwargs.get("index", 1))
        bbox = kwargs.get("bbox_2d")
        
        if not bbox or len(bbox) != 4:
            raise ValueError("Zoom-in requires valid bbox_2d: [x1, y1, x2, y2]")
        
        # Get base image from cache
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img = Image.fromarray(base_arr)
        w, h = img.width, img.height
        
        # Rescale and clamp bbox
        x1, y1, x2, y2 = self._rescale_and_clamp_bbox(bbox, w, h)
        
        # Crop image
        cropped = img.crop((x1, y1, x2, y2))
        cropped_arr = np.array(cropped.convert("RGB"))
        
        # Save cropped image
        new_path = str(self.overlay_dir / f"zoom_{uuid.uuid4().hex}.png")
        cropped.save(new_path)
        
        return self._register_image(
            path=str(Path(new_path).resolve()),
            mask_path=base_image_meta.get("mask_path"),
            arr=cropped_arr
        )
    
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
        
        Supports two formats:
        1. Standard format:
           <tool_call>
           {
               "name": "tool_name",
               "arguments": {...}
           }
           </tool_call>
        
        2. Medical tools format:
           <tool_call>
           SAM2
           ```json
           {
               "index": 1,
               "bbox_2d": [x1, y1, x2, y2]
           }
           ```
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
                
                # Try medical tools format first: Tool name + ```json ... ```
                pattern = r"([^\r\n`]+)\s*```json\s*([\s\S]*?)\s*```"
                match = re.search(pattern, tool_call_str, flags=re.IGNORECASE)
                
                if match:
                    # Medical tools format
                    tool_name = match.group(1).strip()
                    json_part = match.group(2).strip()
                    
                    # Remove JSON comments (limitation: may fail if JSON strings contain /* or */)
                    json_part = self._strip_json_comments(json_part)
                    
                    # Parse JSON arguments
                    arguments = json.loads(json_part)
                    
                    # Handle array or single object
                    # Note: In medical tools format, arrays can contain multiple tool calls,
                    # but we only process the first one per tool invocation
                    if isinstance(arguments, list):
                        if len(arguments) == 0:
                            arguments = {}
                        elif len(arguments) > 1:
                            # Log warning if multiple items present (only first will be used)
                            import warnings
                            warnings.warn(
                                f"Tool call contains {len(arguments)} items in array, "
                                f"but only the first will be processed. "
                                f"Consider making separate tool calls for each item.",
                                UserWarning
                            )
                            arguments = arguments[0]
                        else:
                            arguments = arguments[0]
                    
                    return {
                        "name": tool_name,
                        "arguments": arguments if isinstance(arguments, dict) else {}
                    }
                else:
                    # Standard format: parse as JSON directly
                    tool_call = json.loads(tool_call_str)
                    return tool_call
                    
        except (ValueError, json.JSONDecodeError, IndexError):
            pass
        
        return None
    
    def _strip_json_comments(self, json_str: str) -> str:
        """
        Remove JavaScript-style comments from JSON string.
        
        Args:
            json_str: JSON string potentially containing comments
            
        Returns:
            Clean JSON string
        """
        # Remove multi-line comments /* ... */
        json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
        # Remove single-line comments // ...
        json_str = re.sub(r"//[^\r\n]*", "", json_str)
        # Remove excess whitespace
        json_str = re.sub(r"\s+", " ", json_str).strip()
        return json_str
    
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
        
        For medical tools (SAM2, BiomedParse, Zoom-in), uses the medical agent
        feedback format with image metadata. For other tools, uses generic format.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result from tool execution
            
        Returns:
            Formatted string to append to model context
        """
        # Check if this is a medical tool result
        is_medical_tool = tool_name in ["SAM2", "BiomedParse", "Zoom-in"]
        # Medical tool results contain image metadata; 'path' is required for internal use
        # but only index, width, height are shown in the feedback message
        is_medical_result = isinstance(result, dict) and all(
            k in result for k in ["index", "path", "width", "height"]
        )
        
        if is_medical_tool and is_medical_result:
            # Use medical agent tool feedback format
            return MEDICAL_AGENT_TOOL_FEEDBACK.format(
                image_index=result["index"],
                width=result["width"],
                height=result["height"]
            )
        else:
            # Use generic tool result format for non-medical tools
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
    
    def _ensure_chat_history(self, messages: Dict[str, Any]) -> bool:
        """
        Helper to detect if messages use chat-style history format.
        
        Chat-style history is indicated by having a "messages" key with a list
        of message objects containing "role" and "content" fields.
        
        Args:
            messages: Input messages dictionary
            
        Returns:
            True if messages uses chat-style history, False otherwise
        """
        if "messages" not in messages:
            return False
        
        msg_list = messages.get("messages")
        if not isinstance(msg_list, list) or len(msg_list) == 0:
            return False
        
        # Check if first message has expected structure
        first_msg = msg_list[0]
        return isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg
    
    def generate_output(self, messages: Dict[str, Any]) -> str:
        """
        Generate output with tool calling support.
        
        This method extends the base model's generate_output to support
        tool calls. If the model requests a tool call, it executes the tool
        and feeds the result back to the model.
        
        When medical tools are configured, this method automatically injects
        the medical system prompt to provide tool usage guidance, output format
        requirements, and reasoning tips to the model.
        
        Args:
            messages: Input messages dictionary
            
        Returns:
            Final model response after any tool calls
        """
        self.tool_call_history = []
        current_messages = messages.copy()
        
        # Detect if using chat-style history
        use_chat_history = self._ensure_chat_history(current_messages)
        
        # Inject medical system prompt if medical tools are configured
        if self.medical_tools_config:
            if use_chat_history:
                # For chat history, inject as first system message if not present
                msg_list = current_messages.get("messages", [])
                if not msg_list or msg_list[0].get("role") != "system":
                    current_messages["messages"] = [
                        {"role": "system", "content": MEDICAL_AGENT_SYSTEM_PROMPT}
                    ] + msg_list
            else:
                # For prompt-based, inject system key if not present
                if "system" not in current_messages:
                    current_messages["system"] = MEDICAL_AGENT_SYSTEM_PROMPT
        
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
                
                # Append results based on message format
                if use_chat_history:
                    # For chat history: append model response as assistant message
                    # and tool feedback as user message
                    current_messages["messages"].append({
                        "role": "assistant",
                        "content": response
                    })
                    current_messages["messages"].append({
                        "role": "user",
                        "content": tool_result_str
                    })
                else:
                    # For prompt-based: append to prompt string (existing behavior)
                    # Initialize 'prompt' key if missing (without re-copying messages)
                    if "prompt" not in current_messages:
                        current_messages["prompt"] = f"\nPrevious response: {response}\n"
                    
                    # Append tool result to prompt
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
