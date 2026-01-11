from modules.board import ChessBorard
from modules.model import build_model
from modules.utils import find_best_steps, convert_dot2pixel

if __name__ == "__main__":
    import imageio.v3 as iio
    from pathlib import Path

    chessboard_model = ChessBorard(
        det_model=build_model(r"weights/best.pth")
    )
    im1 = Path(r"C:\Users\sun\Downloads\Screenshot_20260110_211941.jpg") # 手机游戏屏幕截图
    demo1image = iio.imread(im1)
    chessboard, positions = chessboard_model.recognize_chessboard(demo1image)
    rects = find_best_steps(chessboard.tolist()).get("rects", [])

    # 将路径坐标映射到像素并绘制前 3 条
    pixelpaths = convert_dot2pixel(
        demo1image,
        rects,
        chessboard_model.template_position_matrix,
        chessboard_model.W,
        chessboard_model.H,
    )

    import cv2

    for pp in pixelpaths[:3]:
        (x1, y1), (x2, y2) = pp
        cv2.rectangle(demo1image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imwrite("output_test_localim.jpg", demo1image)
    