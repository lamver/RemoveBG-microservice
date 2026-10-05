import os

import onnxruntime as ort
from fastapi import FastAPI, UploadFile, Response
from rembg import remove, new_session

app = FastAPI()

# Модель и расход памяти настраиваются переменными окружения; по умолчанию
# всё как раньше (birefnet-general с резервированием памяти ONNX Runtime).
#
# BG_MODEL — имя модели rembg. Замер 05.10.2026 на 8 ядрах: birefnet-general
# 13 ГБ / 10 с на картинку, birefnet-general-lite почти того же качества —
# 6,8 ГБ / 8 с (с BG_CPU_MEM_ARENA=0), isnet-general-use заметно хуже на людях.
# BG_CPU_MEM_ARENA=0 — не держать память про запас: 13 → 8 ГБ у полной модели.
# Число потоков rembg берёт из OMP_NUM_THREADS.
MODEL = os.getenv("BG_MODEL", "birefnet-general")
CPU_MEM_ARENA = os.getenv("BG_CPU_MEM_ARENA", "1") != "0"

_SessionOptions = ort.SessionOptions


class _Options(_SessionOptions):
    def __init__(self):
        super().__init__()
        self.enable_cpu_mem_arena = CPU_MEM_ARENA


# rembg создаёт SessionOptions сам и не даёт передать свои, поэтому
# подменяем класс до создания сессии.
ort.SessionOptions = _Options
session = new_session(MODEL)

@app.post("/remove-bg")
async def remove_background(file: UploadFile):
    input_data = await file.read()
    
    # КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
    # 1. Выключаем alpha_matting совсем (он главный виновник мыла на тексте)
    # 2. Выключаем post_process_mask (он иногда сглаживает углы букв)
    output_data = remove(
        input_data,
        session=session,
        alpha_matting=False, 
        post_process_mask=True
    )
    
    return Response(content=output_data, media_type="image/png")

@app.get("/helth")
def health_check():
    return {"status": "online", "model": MODEL, "cpu_mem_arena": CPU_MEM_ARENA}
