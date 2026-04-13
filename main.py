from fastapi import FastAPI, UploadFile, Response
from rembg import remove, new_session
import io

app = FastAPI()

# Попробуем модель 'u2netp' (она более "грубая" и меньше мылит края на графике)
session = new_session("birefnet-general")

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
    return {"status": "online"}
