import time

import numpy as np
import pyautogui
import pygetwindow as gw

from modules.board import ChessBorard
from modules.model import build_model
from modules.ocrclicker import OCRClicker
from modules.ppdocr import ocr_recognition
from modules.utils import drag_mouse, find_best_steps, convert_dot2pixel


def main():
    chessboard_model = ChessBorard(
        det_model=build_model("weights/best.pth")
    )
    
    auto=False
    while True:
        if not auto:
            input("请切换到游戏窗口，准备好后按回车继续...")   
        try:
            bot = OCRClicker(
                window_title="开局托儿所",
                ocr_func=ocr_recognition
            )
            
            time.sleep(5)
            img = bot.window_im
            demo1image = np.array(img)
            chessboard, positions = chessboard_model.recognize_chessboard(demo1image)
            print("棋盘分析完成，开始寻找路径...")
            results = find_best_steps(chessboard.tolist(),max_times=45)
            print(f"路径寻找完成，开始执行操作...，共移除数字：{results.get('best_removed', 0)}")
            rects = results.get("rects", [])
            if not rects:
                raise Exception("No paths found")
            pixelpaths = convert_dot2pixel(
                demo1image,
                rects,
                chessboard_model.template_position_matrix,
                chessboard_model.W,
                chessboard_model.H,
            )

            for i, pp in enumerate(pixelpaths):
                print(f"Executing path {i + 1}/{len(pixelpaths)} -- {pp}")
                offset = int(chessboard_model.avgd * 0.3)
                pp = (
                    [pp[0][0] - offset, pp[0][1] - offset],
                    [pp[1][0] + offset, pp[1][1] + offset],
                )
                positions=bot.position
                left, top = positions[0], positions[1]
                drag_mouse(pp, left, top)
                # time.sleep(np.random.rand())
            print("本轮操作完成，等待下一轮...")
            pyautogui.sleep(5)
            while True:
                bot.click(["确定"], timeout=10)
                time.sleep(2)
                print("点击确定，等待下一步...")
                if bot.click(["再来一局","再次挑战"], timeout=5):
                    print("检测到再来一局，开始下一轮操作...")
                    auto=True
                    break
        except Exception as e:
            print(f"出现错误: {e}")
            continue


if __name__ == "__main__":
    main()
