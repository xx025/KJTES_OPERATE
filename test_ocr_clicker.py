from modules.ocrclicker import OCRClicker
from modules.ppdocr import ocr_recognition

bot = OCRClicker(
    window_title="开局托儿所",
    ocr_func=ocr_recognition
)

bot.click(["再来一局","再次挑战"], timeout=5)
print("检测到再来一局，开始下一轮操作...")