import time
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pyautogui
import pygetwindow as gw

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2 (相对窗口)


class OCRClicker:
    def __init__(
        self,
        window_title: str,
        ocr_func: Callable[[Any], List[Dict[str, Any]]],
        *,
        min_score: float = 0.8,
        interval: float = 0.25,
        offset: Point = (0, 0),  # 点击偏移（像素）
        activate: bool = True,
        bbox_key: str = "box",  # 你的 OCR 四元组字段名：box/bbox/points 任选其一
    ):
        self.title = window_title
        self.ocr_func = ocr_func
        self.min_score = float(min_score)
        self.interval = float(interval)
        self.offset = offset
        self.activate = activate
        self.bbox_key = bbox_key
        self._win = None

    # -------- window / screenshot --------
    def _win_obj(self):
        if self._win is None:
            wins = gw.getWindowsWithTitle(self.title)
            if not wins:
                raise RuntimeError(f"未找到窗口: {self.title}")
            self._win = wins[0]
        return self._win

    def _region(self) -> Tuple[int, int, int, int]:
        w = self._win_obj()
        if self.activate:
            try:
                w.activate()
            except Exception:
                pass
        return (w.left, w.top, w.width, w.height)

    # -------- bbox / match --------
    @staticmethod
    def _center(b: BBox) -> Point:
        x1, y1, x2, y2 = b
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _get_bbox(self, it: Dict[str, Any]) -> Optional[BBox]:
        b = it.get(self.bbox_key)
        if not b:
            return None
        x1, y1, x2, y2 = b
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _match(cand: str, tgt: str, mode: str) -> bool:
        cand = cand.strip()
        tgt = tgt.strip()
        if mode == "equals":
            return cand == tgt
        if mode == "contains":
            return tgt in cand
        if mode == "regex":
            return bool(re.search(tgt, cand))
        raise ValueError(f"未知 mode: {mode}")

    @staticmethod
    def _targets(target: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(target, str):
            t = target.strip()
            return [t] if t else []
        out = [str(t).strip() for t in target]
        return [t for t in out if t]

    # -------- click helpers --------
    def _click_rel(self, p_rel: Point):
        left, top, _, _ = self._region()
        x = left + int(p_rel[0]) + self.offset[0]
        y = top + int(p_rel[1]) + self.offset[1]
        pyautogui.click(x=x, y=y)

    def _ocr_hits(self) -> List[Tuple[str, float, BBox]]:
        left, top, w, h = self._region()
        img = pyautogui.screenshot(region=(left, top, w, h))
        raw = self.ocr_func(img) or []

        hits = []
        for it in raw:
            score = float(it.get("score", 0.0))
            if score < self.min_score:
                continue
            text = str(it.get("text", "")).strip()
            b = self._get_bbox(it)
            if text and b is not None:
                hits.append((text, score, b))
        return hits

    # -------- core API --------
    def click(
        self,
        target: Union[str, Sequence[str], Point],
        *,
        timeout: float = 0.0,
        mode: str = "equals",
    ) -> bool:
        """
        唯一核心 API：
          - target: "文字" / ["文字1","文字2"] / (x,y)  (x,y 为相对窗口坐标)
          - timeout=0 不等待；timeout>0 等待出现并点
          - mode: equals / contains / regex
        """
        # 坐标点击：相对窗口
        if isinstance(target, tuple) and len(target) == 2:
            self._click_rel((int(target[0]), int(target[1])))
            return True

        targets = self._targets(target)  # type: ignore[arg-type]
        if not targets:
            return False

        end = time.time() + float(timeout)
        while True:
            hits = self._ocr_hits()

            # 多文本：按 targets 顺序优先；同一 target 里取 score 最高
            for t in targets:
                cand = [
                    (text, score, b)
                    for (text, score, b) in hits
                    if self._match(text, t, mode)
                ]
                if cand:
                    _, _, b = max(cand, key=lambda x: x[1])
                    self._click_rel(self._center(b))
                    return True

            if timeout <= 0 or time.time() >= end:
                return False
            time.sleep(self.interval)

    @property
    def position(self) -> BBox:
        """_summary_

        Returns:
            BBox: _description_
            (w.left, w.top, w.width, w.height)
        """
        return self._region()

    
    @property
    def window_im(self) -> Any:
        left, top, w, h = self._region()
        img = pyautogui.screenshot(region=(left, top, w, h))
        return img