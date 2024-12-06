from pydantic import BaseModel
from model.model import VideoModel
from fastapi import FastAPI

app = FastAPI()
model = VideoModel()

class TextInput(BaseModel):
    text: str

@app.post("/generate-video")
async def generate_video(input: TextInput):
    video = model.generate_wav2lip_video(text=input.text, image_url="https://raw.githubusercontent.com/GDGoC-2024-RE-ALThon-TEAM8/AI_1/refs/heads/main/ai_human.jpg")
    return {"video": video}