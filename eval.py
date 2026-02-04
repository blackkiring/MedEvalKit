from __future__ import annotations
import os
import json
import base64
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets
from PIL import Image
import io
import re
import uuid
import shutil
import requests
from pathlib import Path
from typing import Any, Tuple
from openai import OpenAI

# ============================
# 全局配置（请根据实际环境修改）
# ============================
VLLM_BASE_URL = ""  # vLLM服务地址
VLLM_API_KEY = ""  # 你的API KEY
VLLM_MODEL_NAME = ""  # 你的多模态模型名
MAX_TOKENS = 8192  # 模型最大生成token数
TEMPERATURE = 0.01  # 推理温度
TOP_P = 1.0
SKIP_EXISTED = True  # 是否跳过已生成结果的样本（断点续跑）
OUTPUT_FILE = "mmmu_vllm_result.json"  # 最终结果保存文件
CACHE_DIR = "/vepfs-vpc-mlp2/fs-ift/med/xulin-ustc/mmmu_cache"  # 图片缓存目录
TOOL_SERVER_URL = "http://192.168.11.48:6060"  # 主工具服务器地址
BIOMEDPARSE_URL = "http://192.168.11.48:6061"  # BiomedParse独立地址
MAX_STEPS = 20  # 单样本最大推理步数

# 初始化vLLM客户端
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY
)

# ============================
# 工具函数：图像转Base64（vLLM多模态标准输入）
# ============================
def img2base64(img: np.ndarray) -> str:
    """将numpy数组图像转换为vLLM支持的Base64编码字符串"""
    # 处理灰度图/通道数异常
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
        img = np.stack([img]*3, axis=-1)
    # 转换为PIL Image并编码
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")  # PNG格式无压缩，避免失真
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

# ============================
# 复用agent_system中的辅助函数（原代码依赖）
# ============================
def _read_image_size(img_path: str) -> Tuple[int, int]:
    """读取图像宽高"""
    with Image.open(img_path) as img:
        return img.width, img.height

def _load_image_array(img_path: str) -> np.ndarray:
    """加载图像为numpy数组（RGB格式）"""
    img = Image.open(img_path).convert("RGB")
    return np.array(img)

class MedicalToolClient:
    """工具服务器客户端（原代码依赖）"""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30  # 超时时间30s

    def run(self, endpoint: str, payload: dict, expect_binary: bool = False) -> Any:
        """调用工具接口"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()  # 抛出HTTP错误
            if expect_binary:
                return response.content
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Tool request failed: {str(e)}") from e

# ============================
# 完整MedicalLocalWorker类（适配多工具调用）
# ============================
class MedicalLocalWorker:
    """单进程本地 Worker，支持多工具组合调用"""
    def __init__(self, seed: int = 42, env_kwargs: dict = None):
        self._rng = np.random.RandomState(seed)
        self.env_kwargs = env_kwargs or {}
        
        # 输出配置
        self.output_dir = Path(self.env_kwargs.get("output_dir", "./eval_results_local"))
        self.mask_dir = self.output_dir / "masks"
        self.overlay_dir = self.output_dir / "overlays"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # 工具配置
        self.tool_server_url = self.env_kwargs.get("tool_server_url", TOOL_SERVER_URL)
        self._tool_clients = {
            "SAM2": MedicalToolClient(self.tool_server_url),
            "BiomedParse": MedicalToolClient(self.env_kwargs.get("biomedparse_url", BIOMEDPARSE_URL))
        }
        self.tool_endpoints = {
            "BiomedParse": "/biomedparse",
            "SAM2": "/segment",
        }
        self.DEFAULT_TOOLS = ["SAM2", "BiomedParse", "Zoom-in"]  # 支持的工具列表

        # 状态变量
        self._current_case = None
        self._image_cache = [] # 存储 numpy array（图像缓存，避免重复读盘）
        self._images = []      # 存储图像元信息
        self._action_history = []
        self._step_count = 0
        self._done = False
        self._tool_used = False
        self.max_steps = self.env_kwargs.get("max_steps", MAX_STEPS)

    def reset(self, case_data: dict):
        """重置环境，加载新的 Case"""
        self._current_case = case_data
        self._step_count = 0
        self._done = False
        self._tool_used = False
        self._action_history = []
        self._image_cache = []
        self._images = []
        
        # 1. 加载初始图像
        img_path = case_data["image_path"][0]
        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Image not found: {img_path}")
             
        initial_arr = _load_image_array(img_path)
        self._base_hw = initial_arr.shape[:2]
        
        # 2. 注册图像
        self._register_image(path=img_path, arr=initial_arr)
        
        # 3. 处理 GT Mask (如果有)
        self._gt_mask_cache = None
        if case_data.get("mask_path") and os.path.exists(case_data["mask_path"]):
             self._gt_mask_cache = self._load_mask_array_from_path(case_data["mask_path"], self._base_hw)

        # 4. 返回初始观测 (Prompt + Image)
        raw_obs = {
            "question": case_data["question"],
            "options": case_data.get("options", ""),
            "image_index": 0
        }
        return raw_obs, [initial_arr], self._build_info()

    def step(self, action: str):
        """执行一步动作，支持多工具组合调用"""
        if self._done:
             return "", [], 0.0, True, self._build_info()

        self._step_count += 1
        self._action_history.append(action)
        
        # 1. 解析多工具调用（核心：支持数组格式的多个工具）
        feedback_str = ""
        new_images = []
        tool_invocations = self._parse_tool_invocations(action)
        
        if tool_invocations:
            self._tool_used = True
            # 遍历执行每个工具调用，单独异常处理
            for idx, item in enumerate(tool_invocations):
                tool_name = item["tool"]
                payload_list = item.get("payload", [{}])
                for payload in payload_list:
                    try:
                        # 执行单个工具操作，返回新图像元信息
                        image_meta = self._execute_tool_action(tool_name, payload)
                        feedback_str += f"✅ Tool {tool_name} (idx{idx+1}) success. New img index: {image_meta['index']}\n"
                        new_images.append(self._image_cache[-1])  # 获取新生成的图像数组
                    except Exception as exc:
                        # 单个工具失败不影响其他工具，记录错误信息
                        feedback_str += f"❌ Tool error ({tool_name}): {str(exc)[:100]}\n"
                    print(feedback_str)
        # 2. 提取答案并判断结束（触发条件：提取到答案 或 达到最大步数）
        predicted_answer = self._extract_answer(action)
        done = bool(predicted_answer) or (self._step_count >= self.max_steps)
        self._done = done
        
        # 3. 计算指标（准确率 + IoU，IoU仅当有GT Mask时生效）
        acc_score = 0.0
        iou_score = 0.0
        
        if done:
            # 计算准确率
            target = str(self._current_case.get("answer", "")).upper()
            if predicted_answer and target and predicted_answer == target:
                acc_score = 1.0
            
            # 计算IoU（仅当存在GT Mask且使用过工具时）
            if self._gt_mask_cache is not None and self._tool_used:
                 # 找最后一个生成的Mask用于IoU计算
                 pred_mask_path = None
                 for meta in reversed(self._images):
                     if meta.get("mask_path"):
                         pred_mask_path = meta.get("mask_path")
                         break
                 if pred_mask_path and os.path.exists(pred_mask_path):
                     pred_arr = self._load_mask_array_from_path(pred_mask_path, self._base_hw)
                     pred_bin = pred_arr > 0.5
                     gt_bin = self._gt_mask_cache > 0.5
                     inter = np.logical_and(pred_bin, gt_bin).sum()
                     union = np.logical_or(pred_bin, gt_bin).sum()
                     iou_score = float(inter) / float(union) if union > 0 else 0.0

        # 构造返回信息
        info = self._build_info()
        info["metrics"] = {"accuracy": acc_score, "iou": iou_score, "steps": self._step_count}
        info["prediction"] = predicted_answer
        
        return feedback_str, new_images, 0.0, done, info

    def _register_image(self, path, mask_path=None, arr=None):
        """注册图像到缓存，记录元信息"""
        if arr is None:
            arr = _load_image_array(path)
        width, height = _read_image_size(path)
        self._image_cache.append(arr)
        meta = {
            "index": len(self._images) + 1,  # 图像索引从1开始
            "path": path, 
            "width": width, 
            "height": height, 
            "mask_path": mask_path
        }
        self._images.append(meta)
        return meta

    def _parse_tool_invocations(self, action):
        """
        核心重构：解析多工具调用，支持JSON数组格式
        适配格式：Action: [SAM2/BiomedParse/Zoom-in] ```json [{"index":1,"bbox_2d":[...]}] ```
        返回：[{"tool": "SAM2", "payload": [{}]}, ...]
        """
        if not action or "Action:" not in action:
            return []
        
        # 正则匹配：Action: 工具名 + ```json 内容 ```
        pattern = r"<tool_call>\s*([^\r\n`]+)\s*```json\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, action, flags=re.IGNORECASE)
        if not matches:
            return []
        tool_invocations = []
        for tool_part, json_part in matches:
            # 提取工具名（支持多工具用/分隔，如SAM2/Zoom-in）
            tool_names = [t.strip() for t in tool_part.split("/") if t.strip()]
            tool_names = [t for t in tool_names if t in self.DEFAULT_TOOLS]
            if not tool_names:
                continue
            
            # 去除JSON中的注释，避免解析失败
            clean_json = self._strip_json_comments(json_part)
            if not clean_json:
                continue
            
            # 解析JSON（支持对象/数组，统一转为数组）
            try:
                payload = json.loads(clean_json)
                payload_list = payload if isinstance(payload, list) else [payload]
                # 过滤空payload
                payload_list = [p for p in payload_list if isinstance(p, dict) and p]
            except json.JSONDecodeError:
                continue
            
            # 每个工具单独封装，支持多工具并行调用
            for tool in tool_names:
                tool_invocations.append({
                    "tool": tool,
                    "payload": payload_list
                })
        
        return tool_invocations

    def _strip_json_comments(self, json_str: str) -> str:
        """去除JSON字符串中的//单行注释和/* */多行注释，补全原代码缺失"""
        # 去除多行注释 /* ... */
        json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
        # 去除单行注释 // ...
        json_str = re.sub(r"//[^\r\n]*", "", json_str)
        # 去除多余空白符
        json_str = re.sub(r"\s+", " ", json_str).strip()
        return json_str

    def _get_image_by_index(self, idx: int) -> dict:
        """根据索引获取图像元信息，补全原代码缺失"""
        if idx < 1 or idx > len(self._images):
            raise IndexError(f"Image index {idx} out of range (1-{len(self._images)})")
        return self._images[idx - 1]

    def _execute_tool_action(self, tool, payload):
        """工具执行路由，分发到对应工具的处理方法"""
        tool_handler = {
            "SAM2": self._handle_sam2,
            "BiomedParse": self._handle_biomedparse,
            "Zoom-in": self._handle_zoom_in
        }.get(tool)
        if not tool_handler:
            raise ValueError(f"Unknown tool: {tool}")
        return tool_handler(payload)

    def _handle_sam2(self, payload: dict) -> dict:
        """SAM2工具处理：边界框分割，生成红色掩码叠加图"""
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("SAM2 requires valid bbox_2d: [x1, y1, x2, y2]")

        # 从缓存获取基础图像，避免重复读盘
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img_path = base_image_meta["path"]
        h, w = base_arr.shape[:2]

        # 坐标还原与边界限制（相对1024转像素坐标）
        rescaled_bbox = list(self._rescale_and_clamp_bbox(bbox, w, h))
        extra = {"bbox": rescaled_bbox, "bbox_2d": rescaled_bbox}
        # 追加可选参数
        for k in ["clicklist", "labels", "multimask_output", "return_logits"]:
            if k in payload:
                extra[k] = payload[k]

        # 调用工具服务器获取掩码
        npz_bytes = self._call_segmenter("SAM2", img_path, extra, expect_binary=True)
        
        # 解析掩码，取置信度最高的
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

        # 保存掩码到本地
        mask_path = str(self.mask_dir / f"sam2_mask_{uuid.uuid4().hex}.png")
        Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255).save(mask_path)

        # 内存生成叠加图（红色掩码+原图）
        overlay_arr = self._create_overlay_array(base_arr, best_mask, alpha=0.5)
        overlay_path = str(self.overlay_dir / f"sam2_overlay_{uuid.uuid4().hex}.png")
        Image.fromarray(overlay_arr).save(overlay_path)

        # 注册新图像到缓存（直接传数组，避免后续读盘）
        return self._register_image(
            path=str(Path(overlay_path).resolve()),
            mask_path=str(Path(mask_path).resolve()),
            arr=overlay_arr
        )
        
    def _handle_biomedparse(self, payload: dict) -> dict:
        """BiomedParse工具处理：文本描述分割，生成红色掩码叠加图"""
        idx = int(payload.get("index", 1))
        captions = payload.get("captions", "")
        if not captions or not isinstance(captions, str):
            raise ValueError("BiomedParse requires non-empty captions string")

        # 从缓存获取基础图像
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img_path = base_image_meta["path"]
        h, w = base_arr.shape[:2]
        # 调用工具服务器获取掩码路径
        tool_mask_path = self._call_segmenter("BiomedParse", img_path, {"captions": captions})
        print(tool_mask_path)
        if not tool_mask_path or not os.path.exists(tool_mask_path):
            raise RuntimeError("BiomedParse did not return a valid mask path")

        # 复制掩码到本地输出目录
        mask_dst_path = str(self.mask_dir / f"biomedparse_mask_{uuid.uuid4().hex}.png")
        shutil.copy2(tool_mask_path, mask_dst_path)

        # 加载掩码并生成叠加图
        mask_arr = self._load_mask_array_from_path(mask_dst_path, (h, w))
        overlay_arr = self._create_overlay_array(base_arr, mask_arr, alpha=0.5)
        overlay_path = str(self.overlay_dir / f"biomedparse_overlay_{uuid.uuid4().hex}.png")
        Image.fromarray(overlay_arr).save(overlay_path)

        # 注册新图像
        return self._register_image(
            path=str(Path(overlay_path).resolve()),
            mask_path=str(Path(mask_dst_path).resolve()),
            arr=overlay_arr
        )
    
    def _handle_zoom_in(self, payload: dict) -> dict:
        """Zoom-in工具处理：边界框裁剪，生成裁剪后的图像"""
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("Zoom-in requires valid bbox_2d: [x1, y1, x2, y2]")

        # 从缓存获取基础图像
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img = Image.fromarray(base_arr)
        w, h = img.width, img.height

        # 坐标还原与边界限制
        x1, y1, x2, y2 = self._rescale_and_clamp_bbox(bbox, w, h)
        # 裁剪图像（PIL高效裁剪）
        cropped = img.crop((x1, y1, x2, y2))
        cropped_arr = np.array(cropped.convert("RGB"))

        # 保存裁剪图并注册到缓存
        new_path = str(self.overlay_dir / f"zoom_{uuid.uuid4().hex}.png")
        cropped.save(new_path)
        return self._register_image(
            path=str(Path(new_path).resolve()),
            mask_path=base_image_meta.get("mask_path"),
            arr=cropped_arr
        )

    def _extract_answer(self, action):
        """提取答案，支持<answer>X和Answer:X两种格式"""
        patterns = [
            r"<answer>\s*([A-Za-z])\s*</answer>",
            r"<answer>\s*([A-Za-z])\b",
            r"Action:\s*Answer\s*:\s*([A-Za-z])",
            r"Answer\s*:\s*([A-Za-z])"
        ]
        for pattern in patterns:
            match = re.search(pattern, action, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _build_info(self):
        """构建基础信息字典"""
        return {
            "question_id": self._current_case.get("question_id"),
            "ground_truth": self._current_case.get("answer"),
        }
        
    def _load_mask_array_from_path(self, path, hw):
        """加载掩码并调整为指定尺寸"""
        with Image.open(path) as m:
            m = m.convert("L").resize((hw[1], hw[0]), Image.NEAREST)
            return np.array(m).astype(np.float32) / 255.0

    def _call_segmenter(self, tool, image_path, extra, expect_binary=False):
        """调用工具服务器通用方法"""
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
    
    def _create_overlay_array(self, img_np: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """纯内存生成掩码叠加图（红色），避免重复读盘"""
        h, w = img_np.shape[:2]
        
        # 确保掩码与原图尺寸一致
        if mask.shape != (h, w):
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0

        # 掩码区域染成红色，与原图混合
        mask_bin = mask > 0.5
        overlay = img_np.copy().astype(np.float32)
        overlay[mask_bin] = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        # 混合图像（alpha为掩码透明度）
        vis = (img_np.astype(np.float32) * (1.0 - alpha) + overlay * alpha)
        return vis.clip(0, 255).astype(np.uint8)
    
    def _rescale_and_clamp_bbox(self, bbox: list[float], width: int, height: int) -> Tuple[int, int, int, int]:
        """将相对1024的坐标还原为像素坐标，并限制在图像边界内"""
        x1, y1, x2, y2 = bbox
        # 相对坐标转像素坐标（模型输出为0-1024相对值）
        x1 = (x1 / 1024.0) * width
        y1 = (y1 / 1024.0) * height
        x2 = (x2 / 1024.0) * width
        y2 = (y2 / 1024.0) * height
        # 取整并限制边界（避免越界）
        x1 = max(0, min(int(round(x1)), width))
        y1 = max(0, min(int(round(y1)), height))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))
        # 修正无效边界框（x2<=x1或y2<=y1）
        if x2 <= x1:
            x2 = min(x1 + 10, width)
        if y2 <= y1:
            y2 = min(y1 + 10, height)
        return x1, y1, x2, y2
def my_vlm_inference(prompt: str, images: list[np.ndarray]) -> str:
    """
    调用本地vLLM部署的多模态模型，生成符合Worker要求的动作字符串
    输出格式兼容原逻辑：要么Tool调用（SAM2），要么Answer
    :param prompt: 对话上下文（问题+选项+历史观测）
    :param images: 图像列表（np.ndarray，RGB格式）
    :return: 模型生成的动作字符串（如Action: SAM2 ```json {...}``` 或 Action: Answer: X）
    """
    try:
        # 1. 构造vLLM多模态Chat Completion输入（OpenAI标准格式）
        messages = [
            {
                "role": "system",
                "content": """### Guidance:\nYou are a helpful assistant specialized in medical image analysis. You have access to several tools that help you segment and examine medical images (e.g. highlighting lesions or tumors) to answer questions.\nYour task is to carefully analyze the image and question, use the tools step-by-step, and provide a well-reasoned final answer through tool invocation feedback.\n\n### Available tools:\nYou can use the following three tools to process the images. After each tool usage, you must wait for and analyze the visualization feedback before proceeding.\n\n1. **Zoom-in**\n- Purpose: Zoom in on a specific region of an image by cropping it to a bounding box for detailed inspection. If a mask is provided, the zoomed image will highlight the mask’s contour.\n- Input format: JSON\n```json\n[{\n    \"index\": i, # Image index\n    \"bbox_2d\": [x1, y1, x2, y2]\n}]\n```\n- Output: Generates zoomed areas for visual inspection of the i-th image\n\n2. **BiomedParse**\n- Purpose: Detect and segment a specified object type in the image (e.g. lesion, tumor) using text descriptions for the targets.\n- Input format: JSON\n```json\n[{\n    \"index\": i, # Image index\n    \"description\": \"target_description\"\n}]\n```\n- Output: Generates segmentation masks for target objects of the i-th image\n\n3. **SAM2**\n- Purpose: Detect and Segment an object in the image given a bounding box.\n- Input format: JSON\n```json\n[{\n    \"index\": i, # Image index\n    \"bbox_2d\": [x1, y1, x2, y2]\n}]\n```\n- Output: Generates segmentation masks for target objects of the i-th image\n\n### Required Output Format:\nFor each reasoning step, you must structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: [Zoom-in/BiomedParse/SAM2]\n```json\n[JSON format coordinates or descriptions]\n```\n\nAfter your reasoning and iteratively refine your solution through tool invocation feedback, you should arrive at a final answer and structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: Answer\n<answer> [Your final answer] </answer>\n\n### Please NOTE the following reasoning techniques:\n1. Initial Analysis\n   - Break down the complex problem\n   - Plan your approach\n\n2. Iterative Reasoning for Each Step\n   - Choose appropriate tool\n   - Provide relative coordinates in JSON format (from 0 to 1024)\n   - Observe the tool invocation output\n   - Reflect on the results returned by the tool:\n     * Does the results of the segmentation or zooming reasonable?\n     * Does it align with your reasoning?\n     * What adjustments are needed?\n   - Backtrack and Adjust:\n     * If errors found, backtrack to previous step to modify actions or decisions as needed."""
            },
            {
                "role": "user",
                "content": []  # 多模态内容：文本+图像
            }
        ]
        
        # 2. 添加文本提示到user content
        messages[1]["content"].append({"type": "text", "text": prompt})
        
        # 3. 所有图像转Base64并添加到user content（vLLM多模态标准）
        for img in images:
            b64_str = img2base64(img)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": b64_str}
            })
        
        # 4. 调用vLLM生成结果
        response = client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=1.0,
            stop=None  # 如需截断可添加stop=["\n"]
        )
        
        # 5. 提取并返回生成结果（去除首尾空格，保证格式）
        action = response.choices[0].message.content.strip()
        return action
    
    except Exception as e:
        # 推理异常时兜底返回Answer（避免中断评测）
        print(f"VLLM推理失败，错误信息：{str(e)[:200]}")
        return "Action: Answer: A"

# ============================
# MMMU医学数据集加载（支持全量，保留图片缓存）
# ============================
CACHE_DIR = "/vepfs-vpc-mlp2/fs-ift/med/xulin-ustc/mmmu_cache"
def load_mmmu_data(cache_dir=CACHE_DIR, subsets: list = None):
    """
    加载MMMU多个子集的验证集数据，自动合并并缓存图片
    Args:
        cache_dir: 图片缓存目录
        subsets: 要加载的子集列表，默认["Clinical_Medicine"]
    Returns:
        list: 合并后的所有样本数据，每条含subset字段标识所属子集
    """
    # 设置默认子集，若未传入则加载原有临床医学子集
    if subsets is None:
        subsets = ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health']
    # 校验入参为列表
    ds_list = []
    assert isinstance(subsets, list) and len(subsets) > 0, "subsets必须为非空列表"
    for subset in subsets:
        print(f"开始加载子集: {subset}")
        ds = load_dataset(
            "/vepfs-vpc-mlp2/fs-ift/med/common/data/duomotai_data/MMMU",  # 你的本地MMMU路径
            subset,
            split="validation",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        ds_list.append(ds)
        print(f"子集 {subset} 加载完成，样本数: {len(ds)}")
    
    # 合并所有子集（不变）
    ds_combined = concatenate_datasets(ds_list)
    print(f"所有子集合并完成，总样本数: {len(ds_combined)}")
    
    data_list = []
    os.makedirs(cache_dir, exist_ok=True)
    
    # 遍历处理数据（仅修改图像处理部分）
    for item in tqdm(ds_combined, desc="Processing MMMU Multi-Subset Data (仅取第一张图)"):
        qid = item["id"]
        question = item["question"]
        opts_str = str(item["options"]) if item.get("options") else ""
        answer = item["answer"]
        
        # ===================== 核心修改：多图仅取第一张 =====================
        first_img = item["image_1"] # 强制只取第一张图
        # ====================================================================
        
        # 唯一命名缓存（不变，仅缓存第一张图）
        img_filename = f"{qid}.png"  # 固定img0，标识为第一张图
        img_path = os.path.join(cache_dir, img_filename)
        if not os.path.exists(img_path):
            first_img.save(img_path, format="PNG")
        img_paths = [img_path]  # 始终为长度1的列表，与单图统一
        
        # 构造样本数据（删除多余的image_num，因仅用第一张）
        data_list.append({
            "question_id": qid,
            "subset": subset,  # 子集标识
            "question": question,
            "options": opts_str,
            "image_path": img_paths,  # 仅第一张图的路径，长度恒为1
            "answer": answer,
            "data_type": "vqa"
        })
    
    print(f"\nMMMU数据处理完成！")
    print(f"总样本数：{len(data_list)} | 所有题目均仅加载/使用第一张图")
    print(f"图像缓存至：{cache_dir}（命名规则：子集_题目ID_img0.png）")
    return data_list


# ============================
# 主评测循环（支持全量、断点续跑、实时保存）
# ============================
def run_local_eval(dataset, output_file=OUTPUT_FILE):
    # 初始化本地MedicalLocalWorker（与原逻辑一致）
    worker = MedicalLocalWorker(
        seed=42,
        env_kwargs={
            "output_dir": "./eval_output_vllm",
            "tool_server_url": TOOL_SERVER_URL
        }
    )
    
    # 断点续跑：加载已生成的结果，跳过已处理样本
    results = []
    if SKIP_EXISTED and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {r["question_id"] for r in results}
        dataset = [case for case in dataset if case["question_id"] not in processed_ids]
        print(f"断点续跑：已处理 {len(results)} 条，剩余 {len(dataset)} 条待评测")
    
    # 遍历全量数据集评测
    for i, case in enumerate(tqdm(dataset, desc="Evaluating MMMU with VLLM")):
        case_id = case["question_id"]
        try:
            # 1. 重置Worker环境，加载当前样本
            obs_dict, current_imgs, info = worker.reset(case)
        except Exception as e:
            print(f"样本 {case_id} 重置失败，跳过：{str(e)[:100]}")
            # 记录失败样本，便于后续分析
            results.append({
                "question_id": case_id,
                "metrics": {"accuracy": 0},
                "trajectory": ["Reset Error"],
                "ground_truth": case["answer"],
                "prediction": None,
                "error": str(e)[:200],
                "subset": case.get("subset", "unknown")
            })
            continue
        
        # ===================== 核心新增：为选项自动添加A/B/C/D字母前缀 =====================
        def add_option_letters(raw_options):
            """
            为原始无字母选项添加A)、B)、C)前缀，适配：字符串(分号/逗号分隔)、列表格式
            :param raw_options: 原始obs_dict['options']，支持str/list
            :return: 带字母前缀的标准化选项字符串
            """
            if not raw_options:
                return ""
            # 步骤1：统一转为列表（处理最常见的分号分隔，兼容逗号分隔）
            if isinstance(raw_options, str):
                # 先按分号分割，再按逗号分割（兼容两种分隔符），去空值和首尾空格
                options = [opt.strip() for opt in re.split(r'[;,]', raw_options) if opt.strip()]
            elif isinstance(raw_options, list):
                # 列表格式直接去空值和首尾空格
                options = [opt.strip() for opt in raw_options if opt.strip()]
            else:
                # 非字符串/列表，返回原内容兜底
                return str(raw_options)
            # 步骤2：按顺序添加A)、B)、C)、D)...前缀（ASCII码65对应A，依次递增）
            labeled_options = []
            for idx, opt in enumerate(options):
                letter = chr(65 + idx)  # 0→A,1→B,2→C...
                labeled_options.append(f"{letter}) {opt}")
            # 步骤3：转回换行分隔的字符串（适配Prompt格式，模型更易识别）
            return "\n".join(labeled_options)
        
        # 处理选项，生成带字母的标准化选项
        labeled_options = add_option_letters(obs_dict['options'])
        # ==================================================================================
            
        # 2. 构造初始Prompt（修改：使用带字母的选项）
        current_prompt = f"Question: {obs_dict['question']}\nOptions: {labeled_options}\n"
        
        step = 0
        done = False
        trajectory = []
        # 初始化info，防止未执行step时为空
        current_info = info.copy()
        
        # 3. 多步推理（最多5步，Answer时立即终止，原有逻辑不变）
        while not done:
            step += 1
            # 调用vLLM生成动作
            action = my_vlm_inference(current_prompt, current_imgs)
            trajectory.append(action)
            print(action)
            # 匹配Answer并立即终止推理
            answer_patterns = [r"Action\s*:\s*Answer", r"<answer>\s*([A-Za-z])\s*</answer>"]
            is_answer = any(re.search(p, action, re.IGNORECASE) for p in answer_patterns)
            if is_answer:
                done = True
                # 提取预测答案
                predicted_ans = None
                for pat in [r"<answer>\s*([A-Za-z])\b", r"Answer\s*:\s*([A-Za-z])"]:
                    match = re.search(pat, action, re.IGNORECASE)
                    if match:
                        predicted_ans = match.group(1).upper()
                        break
                # 手动构造info，保证格式统一
                current_info = {
                    "question_id": case_id,
                    "ground_truth": case["answer"],
                    "prediction": predicted_ans,
                    "metrics": {"accuracy": 0, "steps": step}
                }
                continue

            # 未匹配到Answer，执行原有工具调用逻辑
            feedback, new_imgs, _, done, current_info = worker.step(action)
            
            # 更新Prompt和图像（与原逻辑一致）
            if feedback:
                current_prompt += f"\nObservation: {feedback}"
            if new_imgs and isinstance(new_imgs, list):
                current_imgs.extend(new_imgs)
                
        # 4. 收集当前样本结果（原有逻辑不变）
        results.append({
            "question_id": case_id,
            "subset": case.get("subset", "unknown"),
            "metrics": current_info.get("metrics", {"accuracy": 0, "steps": step}),
            "trajectory": trajectory,
            "ground_truth": case["answer"],
            "prediction": current_info.get("prediction")
        })
        
        # 5. 实时保存（每10条/最后一条，原有逻辑不变）
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"已保存 {len(results)} 条结果到 {output_file}")

    # 6. 计算并打印最终准确率（原有逻辑不变）
    valid_results = [r for r in results if "error" not in r]
    total_all = len(results)
    total_valid = len(valid_results)
    
    if total_valid == 0:
        final_acc = 0.0
        subset_acc = {}
    else:
        # 整体准确率
        final_acc = np.mean([r["metrics"].get("accuracy", 0) for r in valid_results])
        # 按子集统计准确率
        subset_results = defaultdict(list)
        for r in valid_results:
            subset_results[r.get("subset", "unknown")].append(r["metrics"].get("accuracy", 0))
        subset_acc = {k: {"num": len(v), "acc": np.mean(v)} for k, v in subset_results.items()}

    # 打印评测总结
    print("="*80)
    print(f"MMMU评测完成（自动加选项字母+Answer立即终止）！")
    print(f"总样本数：{total_all} | 有效评测：{total_valid} | 失败样本：{total_all - total_valid}")
    print(f"整体平均准确率：{final_acc:.4f} ({final_acc:.2%})")
    # 打印各子集准确率
    if subset_acc:
        print("\n【各子集准确率统计】")
        for subset, info in subset_acc.items():
            print(f"{subset:20s} | 样本数：{info['num']:4d} | 准确率：{info['acc']:.4f} ({info['acc']:.2%})")
    print("="*80)

if __name__ == "__main__":
    # 1. 加载MMMU医学全量验证集（去掉[:5]，如需测试可保留）
    mmmu_dataset = load_mmmu_data()
    # 2. 运行vLLM版评测
    run_local_eval(mmmu_dataset)