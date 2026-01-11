import base64
import json
import io
from pathlib import Path
import requests
import numpy as np
from PIL import Image


def _image_to_base64(image_input) -> str:
    """
    输入标准化 -> 转纯 base64 字符串
    支持: 路径(str/Path), PIL.Image, numpy.ndarray
    """
    img_bytes = None
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {image_input}")
        img_bytes = p.read_bytes()
    elif isinstance(image_input, np.ndarray):
        try:
            pil_img = Image.fromarray(image_input)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Numpy 数组转换失败: {e}")
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        try:
            image_input.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
        except Exception as e:
            raise ValueError(f"PIL Image 转换失败: {e}")
    else:
        raise ValueError(f"不支持的输入类型: {type(image_input)}")
    if not img_bytes:
        raise ValueError("图像数据为空")
    b64 = base64.b64encode(img_bytes).decode("ascii").strip()
    missing = (-len(b64)) % 4
    if missing:
        b64 += "=" * missing
    return b64


def _parse_ocr_result(result: dict) -> list:
    try:
        pr = (
            result.get("result", {})
                  .get("ocrResults", [{}])[0]
                  .get("prunedResult", {})
        )
        texts = pr.get("rec_texts", []) or []
        scores = pr.get("rec_scores", []) or []
        boxes = pr.get("rec_boxes", []) or []
        out = []
        for i, text in enumerate(texts):
            out.append({
                "text": text,
                "score": scores[i] if i < len(scores) else 0.0,
                "box": boxes[i] if i < len(boxes) else [],
            })
        return out
    except Exception:
        return []


def ocr_recognition(image_input, api_url: str = "http://127.0.0.1:8080/ocr") -> list:
    """
    OCR 识别统一接口
    
    Args:
        image_input: 图片输入，支持 文件路径/PIL对象/numpy数组
        api_url: API 地址
        
    Returns:
        识别结果列表: [{'text': '...', 'score': 0.99, 'box': [...]}, ...]
    """
    try:
        b64 = _image_to_base64(image_input)
        payload = {
            "file": b64,
            "fileType": 1,
            "visualize": False
        }
        resp = requests.post(api_url, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"API Error: {resp.status_code}, {resp.text[:200]}")
            return []
        return _parse_ocr_result(resp.json())
        
    except Exception as e:
        print(f"OCR Recognition Failed: {e}")
        return []


