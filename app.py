from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
from langchain_google_genai import ChatGoogleGenerativeAI

from operator import itemgetter
from langchain.schema import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
# from langchain.chat_models import ChatGooglePalm

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 1. 한국어 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# 2. 음성을 텍스트로 변환하는 함수
def speech_to_text(audio_path):
    # 2-1. 오디오 파일 로드
    audio, sr = librosa.load(audio_path, sr=16000)  # 샘플링 레이트 16kHz로 맞춤

    # 2-2. 모델 입력 데이터 생성
    input_values = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).input_values

    # 2-3. 모델 예측
    with torch.no_grad():
        logits = model(input_values).logits

    # 2-4. 텍스트 디코딩
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 3. 텍스트 정규화 함수
def normalize_text(text):
     corrections = {
         "스소은": "해커톤은",
     }
     for wrong, correct in corrections.items():
         text = text.replace(wrong, correct)
     return text

if __name__ == "__main__":
    audio_file = "/content/해커톤은재미있다.m4a"  # 처리할 오디오 파일 경로

    # 음성 텍스트 변환
    raw_result = speech_to_text(audio_file)

    # 텍스트 정규화
    processed_result = normalize_text(raw_result)

    print(processed_result)

API_KEY = 'AIzaSyDJe68pRn5j9MLo2vQp76SBEKP-IFEFpkc'




# LLM 인스턴스 생성
llm = ChatGoogleGenerativeAI(
    api_key=API_KEY,
    model="gemini-1.5-flash",
    temperature=0.7
)


### EDIT HERE ###
prompt1 = ChatPromptTemplate.from_template(
    '''이 모델은 어눌한 한국어 발음을 기반으로 올바른 발음이 뭘지 추측하는 일을 해
    {raw_tts}을 입력으로 받았을 때, 올바른 문장이 무엇일지 추측해줘
    출력으로는 반드시 해당 문장만을 출력해줘'''
)
chain1 = (prompt1
  | llm
  | StrOutputParser()
)


prompt2 = ChatPromptTemplate.from_template(
    """이 모델은 한국어 발음 교정에 대해 전문적인 답변을 제공합니다. 당신은 틀린 음절을 찾고 해당 음절에 대하여 어떻게 수정해야 하는지 답변해야 합니다.

     {right_sentence}를 {raw_tts}로 발음했을 때 어떻게 개선하면 좋을지 알려줘

      다음과 같은 json schema를 사용해줘

        "올바른 문장": str,
        "tts 문장": str,
        "틀린 음절": [
          "음절1": str,
          "발음 수정": str,
          "개선 방법": str
           ],
           "음절2": [
              ...
           ]
        ]


      발음 수정으로는 각 음절 별로 발음 오류를 수정하고 그 결과를 출력해줘
      개선 방법으로는 이를 개선하기 위해 입모양 및 혀모양을 어떻게 해야 하는지 알려줘
      """

)
# chain2 정의
chain2 = (
    {"right_sentence": itemgetter("right_sentence"), "raw_tts": itemgetter("raw_tts")}
    | prompt2
    | llm
    | StrOutputParser()
)



import json

def pronunciation_correction(input_data) -> str:
    right_sentence_value = input_data.get("right_sentence")

    if right_sentence_value is None:
        right_sentence_value = chain1.invoke({"raw_tts": input_data["raw_tts"]})

    max_attempts = 5
    attempt = 0
    result_json = None

    while attempt < max_attempts:
        # chain2 호출
        response = chain2.invoke({
            "raw_tts": input_data["raw_tts"],
            "right_sentence": right_sentence_value
        })

        # 불필요한 ```json과 ``` 제거
        result = response.replace("```json", "").replace("```", "")

        # JSON 변환 시도
        try:
            result_json = json.loads(result)
            return result
        except json.JSONDecodeError:
            attempt += 1
            continue

    return {"error": "Invalid JSON format after 3 attempts"}


json_data = {"raw_tts": processed_result, "right_sentence": "나는 학교에 가고 싶어" }

result = pronunciation_correction(json_data)

print(result)



