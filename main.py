from pathlib import Path
import time

import numpy as np
import pyautogui
import pygetwindow as gw
import json
from modules.board import ChessBorard
from modules.model import build_model
from modules.ocrclicker import OCRClicker
from modules.ppdocr import ocr_recognition
from modules.utils import drag_mouse, find_best_steps, convert_dot2pixel
import imageio.v3 as iio


def no_ad(bot: OCRClicker):
    """检测并关闭广告弹窗"""

    while True:
        if "免广告" in bot:
            print("检测到免广告，开始点击免广告...")
            bot.click(["免广告"], timeout=5, mode="contains")
        time.sleep(5)
        print("等待广告结束...")
        if "已获得奖励" in bot:
            bot.click("关闭", timeout=5, mode="contains")
        time.sleep(5)
        if "开始游戏" in bot:
            print("广告结束，返回游戏界面")
            break


def play_game(bot, chessboard_model, save_board_dir):

    demo1image = np.array(bot.window_im)
    chessboard, positions = chessboard_model.recognize_chessboard(demo1image)

    save_dir = save_board_dir / f"{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(save_dir / "board.png", demo1image)
    # 保存为 npz 文件
    np.savetxt(save_dir / "chessboard.txt", chessboard, fmt="%d")

    print("棋盘分析完成，开始寻找路径...")
    results = find_best_steps(
        chessboard.tolist(), max_times=45, url="http://10.8.0.3:9080/search"
    )
    print("路径寻找结果：", json.dumps(results, ensure_ascii=False, indent=4))

    with open(save_dir / "search.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(
        f"路径寻找完成，开始执行操作...，共移除数字：{results.get('best_removed', 0)}"
    )
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
        positions = bot.position
        left, top = positions[0], positions[1]
        drag_mouse(pp, left, top)


def main():
    chessboard_model = ChessBorard(det_model=build_model("weights/best.pth"))
    save_board_dir = Path("./runs/board/")
    save_board_dir.mkdir(parents=True, exist_ok=True)
    while True:
        bot = OCRClicker(
            window_title="开局托儿所",
            ocr_func=lambda x: ocr_recognition(x, "http://10.8.0.3:8080/ocr"),
        )
        try:
            
            if "无限挑战次数" in bot:
                print("检测到无限挑战次数，开始点击无限挑战次数...")
                bot.click(["无限挑战次数"], timeout=5, mode="contains")
                time.sleep(5)
                no_ad(bot)
                
            if "再次挑战" in bot or "再来一局" in bot or "开始游戏" in bot:
                print("检测到再来一局，开始下一轮操作...")
                bot.click(["再来一局", "再次挑战","开始游戏"], timeout=5)
                time.sleep(5)
                play_game(bot, chessboard_model, save_board_dir)
                continue
            if "确定" in bot:
                print("检测到确定，开始点击确定...")
                bot.click(["确定"], timeout=5)
            if "领取" in bot:
                print("检测到领取，开始点击领取...")
                bot.click(["领取"], timeout=5)
            if "脑细胞不足" in bot:
                print("检测到脑细胞不足，需要获取时间")
                bot.click("谢谢", timeout=5, mode="contains")
                print("返回主界面...")
                bot.click("返回主界面", timeout=5, mode="contains")

        except Exception as e:
            print(f"出现错误: {e}")
            continue


if __name__ == "__main__":
    main()
