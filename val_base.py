from __future__ import annotations
import json
import os
import re
import uuid
import shutil
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Tuple

# === 假设这些常量和 Prompt 依然从你的 agent_system 导入 ===
from agent_system.environments.prompts.medical_agent import (
    MEDICAL_AGENT_USER_PROMPT,
    MEDICAL_AGENT_TOOL_FEEDBACK,
)
# 复用辅助函数
from agent_system.environments.medical_agent_env import (
    MedicalToolClient, 
    _read_image_size, 
    _load_image_array
)

class MedicalLocalWorker:
    """
    单进程本地 Worker，用于串行评测。
    """
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
        self.tool_server_url = self.env_kwargs.get("tool_server_url", "http://192.168.11.48:6060")
        self._tool_clients = {
            "SAM2": MedicalToolClient(self.tool_server_url), # 假设 endpoints 统一
            "BiomedParse": MedicalToolClient(self.env_kwargs.get("biomedparse_url", "http://192.168.11.48:6061"))
        }
        self.tool_endpoints = {
            "BiomedParse": "/biomedparse",
            "SAM2": "/segment",
        }

        # 状态变量
        self._current_case = None
        self._image_cache = [] # 存储 numpy array
        self._images = []      # 存储 meta info
        self._action_history = []
        self._step_count = 0
        self._done = False
        self._tool_used = False
        self.max_steps = self.env_kwargs.get("max_steps", 5)

    def reset(self, case_data: dict):
        """
        重置环境，加载新的 Case。
        Args:
            case_data: 包含 question, options, image_path, answer 的字典
        """
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
        # 这里��回纯文本 Prompt 字典，Runner 负责组装
        raw_obs = {
            "question": case_data["question"],
            "options": case_data.get("options", ""),
            "image_index": 0
        }
        return raw_obs, [initial_arr], self._build_info()

    def step(self, action: str):
        """执行一步动作"""
        if self._done:
             return "", [], 0.0, True, self._build_info()

        self._step_count += 1
        self._action_history.append(action)
        
        # 1. 解析和执行工具
        feedback_str = ""
        new_images = []
        tool_invocations = self._parse_tool_invocations(action)
        
        if tool_invocations:
            self._tool_used = True
            for item in tool_invocations:
                try:
                    payload = item.get("payload", [{}])[0]
                    image_meta = self._execute_tool_action(item["tool"], payload)
                    feedback_str += f"Tool {item['tool']} success. New img index: {image_meta['index']}\n"
                    new_images.append(self._image_cache[-1])
                except Exception as exc:
                    feedback_str += f"Tool error ({item['tool']}): {str(exc)}\n"

        # 2. 提取答案并判断结束
        predicted_answer = self._extract_answer(action)
        done = bool(predicted_answer) or (self._step_count >= self.max_steps)
        self._done = done
        
        # 3. 计算指标
        acc_score = 0.0
        iou_score = 0.0
        
        if done:
            target = str(self._current_case.get("answer", "")).upper()
            if predicted_answer and target and predicted_answer == target:
                acc_score = 1.0
            
            # 简单的 IoU 计算逻辑 (仅当有 GT mask 时)
            if self._gt_mask_cache is not None:
                 # 找最后一个 mask
                 pred_mask_path = None
                 for meta in reversed(self._images):
                     if meta.get("mask_path"):
                         pred_mask_path = meta.get("mask_path")
                         break
                 if pred_mask_path:
                     pred_arr = self._load_mask_array_from_path(pred_mask_path, self._base_hw)
                     pred_bin = pred_arr > 0.5
                     gt_bin = self._gt_mask_cache > 0.5
                     inter = np.logical_and(pred_bin, gt_bin).sum()
                     union = np.logical_or(pred_bin, gt_bin).sum()
                     iou_score = float(inter) / float(union) if union > 0 else 0.0

        info = self._build_info()
        info["metrics"] = {"accuracy": acc_score, "iou": iou_score, "steps": self._step_count}
        info["prediction"] = predicted_answer
        
        return feedback_str, new_images, 0.0, done, info

    # === 下面是直接复用原 Worker 的辅助方法，去掉了 Ray 依赖 ===
    # 为了节省篇幅，这里列出需要保留的方法名，具体实现直接 copy 原来的即可
    
    def _register_image(self, path, mask_path=None, arr=None):
        # 逻辑同原代码：缓存 arr，生成 meta，append 到 self._images
        width, height = _read_image_size(path)
        if arr is None: arr = _load_image_array(path)
        self._image_cache.append(arr)
        meta = {"index": len(self._images)+1, "path": path, "width": width, "height": height, "mask_path": mask_path}
        self._images.append(meta)
        return meta

    def _parse_tool_invocations(self, action):
        # 逻辑同原代码：正则解析 Action
        # ... (略，直接复制) ...
        # 注意：这里需要 _strip_json_comments 和 _current_case['available_tools'] 的处理
        # 简化起见，可以直接假设 DEFAULT_TOOLS
        if not action: return []
        m = re.search(r"Action\s*:\s*([^\r\n]+)", action, flags=re.IGNORECASE)
        if not m: return []
        tool = m.group(1).strip()
        if tool.lower() == "answer": return []
        # ... JSON 解析逻辑 ...
        return [{"tool": tool, "payload": []}] # 简化示例

    def _execute_tool_action(self, tool, payload):
        # 路由到 _handle_sam2 / _handle_biomedparse
        if tool == "SAM2": return self._handle_sam2(payload)
        if tool == "BiomedParse": return self._handle_biomedparse(payload)
        if tool == "Zoom-in": return self._handle_zoom_in(payload)
        raise ValueError(f"Unknown tool: {tool}")

    def _handle_sam2(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("SAM2 requires bbox_2d with four coordinates.")

        # 1. 关键优化：从缓存获取 Base 图像，并获取其路径用于工具请求
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1]
        img_path = base_image_meta["path"]
        
        # 坐标还原 (使用缓存的宽高)
        h, w = base_arr.shape[:2]
        rescaled_bbox = list(self._rescale_and_clamp_bbox(bbox, w, h))
        clicklist = payload.get("clicklist", None)
        labels = payload.get("labels", None)
        # 2. 工具服务器调用
        extra = {"bbox": rescaled_bbox, "bbox_2d": rescaled_bbox}
        if clicklist is not None:
            extra["clicklist"] = clicklist
        if labels is not None:
            extra["labels"] = labels
        if "multimask_output" in payload:
            extra["multimask_output"] = payload["multimask_output"]
        if "return_logits" in payload:
            extra["return_logits"] = payload["return_logits"]

        npz_bytes = self._call_segmenter("SAM2", img_path, extra, expect_binary=True)
        
        # 3. 解析 Mask
        with np.load(io.BytesIO(npz_bytes)) as data:
            masks = data.get("masks")
            scores = data.get("scores")
        
        # 获取最匹配的 mask
        if masks.ndim == 3:
            best_idx = int(np.argmax(scores)) if scores is not None else 0
            best_mask = masks[best_idx].astype(np.float32)
        else:
            best_mask = masks.astype(np.float32)

        # 保存二进制 Mask 到磁盘
        mask_path = str(self.mask_dir / f"sam2_mask_{uuid.uuid4().hex}.png")
        Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255).save(mask_path)

        # 4. 关键优化：生成 Overlay。
        # 这里需要稍微修改 _save_overlay_image 使其支持传入 base_arr 而非 path
        overlay_path = str(self.overlay_dir / f"sam2_overlay_{uuid.uuid4().hex}.png")
        overlay_arr = self._create_overlay_array(base_arr, best_mask, alpha=0.5)
        
        # 保存可视化结果到磁盘供后期检查
        Image.fromarray(overlay_arr).save(overlay_path)

        # 5. 关键优化：将生成好的 Overlay 数组直接注册
        image_meta = self._register_image(
            path=str(Path(overlay_path).resolve()), 
            mask_path=str(Path(mask_path).resolve()), 
            arr=overlay_arr
        )
        return image_meta
        
    def _handle_biomedparse(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        captions = payload.get("captions", "")
        
        # 1. 关键优化：从缓存中获取 Base 图像数组，避免磁盘读取
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1] 
        img_path = base_image_meta["path"]
        h, w = base_arr.shape[:2]

        # 2. 调用工具服务器 (保持网络调用，因为这是外部工具)
        # 注意：这里仍需传 img_path，因为远程服务器需要路径来读取图片
        tool_mask_path = self._call_segmenter("BiomedParse", img_path, {"captions": captions})
        if not tool_mask_path:
            raise RuntimeError("BiomedParse did not return a mask path.")
        tool_mask_path = os.path.abspath(tool_mask_path)

        # 3. 关键优化：整理 Mask 路径，仅执行一次磁盘操作（Move/Copy）
        try:
            mask_dst_path = str(self.mask_dir / f"biomedparse_mask_{uuid.uuid4().hex}.png")
            shutil.copy2(tool_mask_path, mask_dst_path)
            mask_dst_path = str(Path(mask_dst_path).resolve())
        except Exception as exc:
            print(f"!!(Warning) copy mask failed: {exc}")
            mask_dst_path = tool_mask_path

        # 4. 关键优化：内存生成 Overlay 数组
        try:
            # 读取工具生成的掩码并 Resize 到 base 尺寸
            mask_arr = self._load_mask_array_from_path(mask_dst_path, (h, w))
            
            # 使用内存中的 base_arr 和新生成的 mask_arr 进行混合
            # 使用我们在 SAM2 优化中定义的 _create_overlay_array 方法
            overlay_arr = self._create_overlay_array(base_arr, mask_arr, alpha=0.5)
            
            # 保存可视化结果到磁盘（用于记录）
            overlay_path = str(self.overlay_dir / f"biomedparse_overlay_{uuid.uuid4().hex}.png")
            Image.fromarray(overlay_arr).save(overlay_path)
            
            # 5. 关键优化：注册时直接传入生成好的数组
            image_meta = self._register_image(
                path=str(Path(overlay_path).resolve()), 
                mask_path=mask_dst_path, 
                arr=overlay_arr
            )
        except Exception as exc:
            print(f"!!(Warning) overlay failed for BiomedParse: {exc}")
            # 回退方案：如果叠加失败，直接注册掩码
            image_meta = self._register_image(mask_dst_path, mask_path=mask_dst_path)

        return image_meta
    
    def _handle_zoom_in(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("Zoom-in requires bbox_2d with four coordinates.")

        # 1. 关键优化：从缓存中获取 Base 图像数组，避免磁盘读取
        base_image_meta = self._get_image_by_index(idx)
        base_arr = self._image_cache[idx - 1] 
        
        # 2. 将数组转回 PIL 进行裁剪操作（PIL 的裁剪效率很高）
        img = Image.fromarray(base_arr)
        w, h = img.width, img.height
        
        # 坐标还原
        x1, y1, x2, y2 = self._rescale_and_clamp_bbox(bbox, w, h)
        
        # 裁剪并保存
        cropped = img.crop((x1, y1, x2, y2))
        
        # 3. 关键优化：裁剪后的数组直接保存，用于 register_image
        cropped_arr = np.array(cropped.convert("RGB"))
        
        new_path = str(self.overlay_dir / f"zoom_{uuid.uuid4().hex}.png")
        cropped.save(new_path) # 仅为了保存记录

        # 4. 传入裁剪后的数组，避免后续 Rollout 再次读取磁盘
        image_meta = self._register_image(
            path=str(Path(new_path).resolve()), 
            mask_path=base_image_meta.get("mask_path"), # 继承之前的 mask (可选)
            arr=cropped_arr
        )
        return image_meta

    def _extract_answer(self, action):
        # 逻辑同原代码：正则提取 <answer> A
        for pattern in [r"<answer>\s*([A-Za-z])\b", r"Answer\s*:\s*([A-Za-z])"]:
            match = re.search(pattern, action, flags=re.IGNORECASE)
            if match: return match.group(1).upper()
        return None

    def _build_info(self):
        return {
            "question_id": self._current_case.get("question_id"),
            "ground_truth": self._current_case.get("answer"),
        }
        
    def _load_mask_array_from_path(self, path, hw):
        # 复用原逻辑
        with Image.open(path) as m:
            m = m.convert("L").resize((hw[1], hw[0]), Image.NEAREST)
            return np.array(m).astype(np.float32) / 255.0

    def _call_segmenter(self, tool, image_path, extra, expect_binary=False):
        # 复用原逻辑: requests.post
        endpoint = self.tool_endpoints.get(tool)
        client = self._tool_clients.get(tool)
        payload = {"image": image_path, "image_path": image_path}
        payload.update(extra)
        response = client.run(endpoint, payload, expect_binary)
        if isinstance(response, bytes): return response
        return response.get("mask_path") or response.get("image_path")
    
    def _create_overlay_array(self, img_np: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """纯内存计算 Overlay 数组"""
        h, w = img_np.shape[:2]
        
        # 确保 Mask 尺寸一致
        if mask.shape != (h, w):
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0

        mask_bin = mask > 0.5
        overlay = img_np.copy().astype(np.float32)
        # 将 Mask 区域染成红色 [R, G, B]
        overlay[mask_bin] = np.array([255.0, 0.0, 0.0], dtype=np.float32)

        # 混合原图与红色遮罩
        vis = (img_np.astype(np.float32) * (1.0 - alpha) + overlay * alpha)
        return vis.clip(0, 255).astype(np.uint8)
    
    def _rescale_and_clamp_bbox(self, bbox: list[float], width: int, height: int) -> Tuple[int, int, int, int]:
        """将 1024 相对坐标还原为图像像素坐标并限制边界"""
        x1, y1, x2, y2 = bbox
        
        # 1. 还原坐标
        x1 = (x1 / 1024.0) * width
        y1 = (y1 / 1024.0) * height
        x2 = (x2 / 1024.0) * width
        y2 = (y2 / 1024.0) * height
        
        # 2. 限制边界并取整
        x1 = max(0, min(int(round(x1)), width))
        y1 = max(0, min(int(round(y1)), height))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))
        
        # 3. 校验有效性 (防止模型输出反向 bbox)
        if x2 <= x1 or y2 <= y1:
            # 如果无效，尝试修正或报错
            print(f"⚠️ 警告: 还原后的 bbox 无效 ({x1, y1, x2, y2})，将使用微小偏移量。")
            x2 = min(x1 + 10, width)
            y2 = min(y1 + 10, height)
            
        return x1, y1, x2, y2
    
    def _get_image_by_index(self, idx):
        return self._images[idx-1]