from fastapi import FastAPI, UploadFile, Response
from rembg import remove
import io

app = FastAPI()

@app.post("/remove-bg")
async def remove_background(file: UploadFile):
    # Читаем входящий файл
    input_data = await file.read()
    
    # Удаляем фон
    output_data = remove(input_data)
    
    # Возвращаем результат в формате PNG
    return Response(content=output_data, media_type="image/png")

@app.get("/helth")
def health_check():
    return {"status": "online"}
