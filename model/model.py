# 텍스트 가지고 입모양 영상 생성

from IPython.display import HTML, Audio
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import cv2
import subprocess
import requests

from gtts import gTTS
import os

from pydub import AudioSegment


class VideoModel:
    def __init__(
        self, wav2lip_dir="Wav2Lip", checkpoint_path="checkpoints/wav2lip.pth"
    ):
        self.wav2lip_dir = wav2lip_dir
        self.checkpoint_path = checkpoint_path

    def download_image(self, image_url, save_path):
        """
        Download an image from a URL and save it to a local path.

        :param image_url: URL of the image
        :param save_path: Path to save the downloaded image
        """
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return save_path
        else:
            raise ValueError(f"Failed to download image from URL: {image_url}")

    def generate_tts(self, text, output_path):
        tts = gTTS(text=text, lang="ko")  # 한국어 설정
        tts.save(output_path)

        return output_path

    def get_audio_length(self, file_path):
        audio = AudioSegment.from_file(file_path)
        length_in_seconds = len(audio) / 1000  # 길이를 밀리초에서 초로 변환
        return length_in_seconds

    def create_video_from_image(self, image_path, output_video_path, video_length, fps):
        """
        주어진 사진을 반복하여 특정 길이의 영상으로 만드는 함수.

        :param image_path: 입력 사진 파일 경로
        :param output_video_path: 출력 영상 파일 경로
        :param video_length: 영상 길이 (초 단위)
        :param fps: 초당 프레임 수
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # 이미지 크기 가져오기
        height, width, _ = image.shape

        # 동영상 코덱 설정 (MP4V 사용)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 동영상 작성기 초기화
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 필요한 총 프레임 수 계산
        total_frames = int(video_length * fps)

        # 영상에 이미지 반복 작성
        for _ in range(total_frames):
            video_writer.write(image)

        # 동영상 작성기 종료
        video_writer.release()

        return output_video_path

    def generate_wav2lip_video(
        self,
        text,
        image_path=None,
        image_url=None,
        output_path="results/result_voice.mp4",
        temp_dir="temp",
        fps=30,
    ):
        """
        Run the Wav2Lip model to generate a video with synced lip movements.

        :param face_video_path: Path to the input face video
        :param audio_path: Path to the input audio file
        :param output_path: Path to save the output video
        :return: Path to the generated video
        """
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        try:
            # Step 1: Handle Image Input
            if image_url and not image_path:
                image_path = os.path.join(temp_dir, "temp_image.jpg")
                self.download_image(image_url, image_path)
            elif not image_path:
                raise ValueError("Either 'image_path' or 'image_url' must be provided.")

            # Step 2: Generate TTS audio
            tts_audio_path = os.path.join(temp_dir, "tts_audio.mp3")
            self.generate_tts(text, tts_audio_path)

            # Step 3: Get audio length
            audio_length = self.get_audio_length(tts_audio_path)

            # Step 4: Create a video from the static image
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            self.create_video_from_image(
                image_path, temp_video_path, video_length=audio_length, fps=fps
            )

            # Step 5: Run Wav2Lip to synchronize audio with video
            command = [
                "python",
                os.path.join(self.wav2lip_dir, "inference.py"),
                "--checkpoint_path",
                self.checkpoint_path,
                "--face",
                temp_video_path,
                "--audio",
                tts_audio_path,
                "--outfile",
                output_path,
            ]

            subprocess.run(command, check=True)

            return output_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error occurred during Wav2Lip execution: {e}")
        except Exception as e:
            raise e
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
