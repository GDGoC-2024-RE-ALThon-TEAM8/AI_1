from pydub import AudioSegment

# FFmpeg 경로 설정
AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"

from pydantic import BaseModel
from model.model import VideoModel
from fastapi import FastAPI, File, UploadFile

from new_model import speech_to_text, normalize_text

app = FastAPI()
model = VideoModel()


class TextInput(BaseModel):
    text: str


@app.post("/generate-video")
async def generate_video(input: TextInput):
    video = model.generate_wav2lip_video(
        text=input.text,
        image_url="https://raw.githubusercontent.com/GDGoC-2024-RE-ALThon-TEAM8/AI_1/refs/heads/main/ai_human.jpg",
    )
    return video


# 음성을 텍스트로 변환
@app.post("/generate-feedback")
async def generate_feedback(audio_file: UploadFile = File(...)):
    # 음성 텍스트 변환
    raw_result = await speech_to_text(audio_file)

    # 텍스트 정규화
    processed_result = normalize_text(raw_result)

    return processed_result
