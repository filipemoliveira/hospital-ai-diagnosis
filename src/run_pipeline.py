from pathlib import Path
import json
import time
import os

import cv2
from ultralytics import YOLO

from pydub import AudioSegment

import azure.cognitiveservices.speech as speechsdk

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES AZURE
# ============================================================

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

LANGUAGE_KEY = os.getenv("AZURE_LANG_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANG_REGION")


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path(
    "../dados/Fase 4/video/vio_1.mp4"
)

AUDIO_PATH = Path(
    "../dados/Fase 4/audio/1.wav"
)

AUDIO_PROCESSED_PATH = Path(
    "../dados/Fase 4/audio/processed_1.wav"
)


# ============================================================
# NORMALIZAÇÃO DE ÁUDIO
# ============================================================

def normalize_audio(input_path: Path, output_path: Path):

    audio = AudioSegment.from_file(str(input_path))

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)

    audio.export(
        str(output_path),
        format="wav"
    )

    return output_path


# ============================================================
# ANÁLISE DE VÍDEO
# ============================================================

def analyze_video(video_path: Path):

    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(str(video_path))

    frame_count = 0

    agitation_scores = []
    closed_posture_count = 0

    previous_keypoints = None

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(frame)

        if (
            len(results) == 0
            or results[0].keypoints is None
            or results[0].keypoints.xy is None
            or len(results[0].keypoints.xy) == 0
        ):
            continue

        # ====================================================
        # KEYPOINTS DA PRIMEIRA PESSOA DETECTADA
        # ====================================================

        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        # ====================================================
        # MOVIMENTO ENTRE FRAMES
        # ====================================================

        if previous_keypoints is not None:

            movement = abs(
                keypoints - previous_keypoints
            ).mean()

            agitation_scores.append(movement)

        previous_keypoints = keypoints

        # ====================================================
        # POSTURA CORPORAL
        # ====================================================

        # 5 = left shoulder
        # 6 = right shoulder
        # 9 = left wrist
        # 10 = right wrist

        left_shoulder_x = keypoints[5][0]
        right_shoulder_x = keypoints[6][0]

        left_wrist_x = keypoints[9][0]
        right_wrist_x = keypoints[10][0]

        shoulder_distance = abs(
            left_shoulder_x - right_shoulder_x
        )

        wrist_distance = abs(
            left_wrist_x - right_wrist_x
        )

        # braços próximos ao corpo
        if wrist_distance < shoulder_distance * 0.6:
            closed_posture_count += 1

    cap.release()

    if frame_count == 0:
        raise Exception("Nenhum frame foi processado.")

    # ========================================================
    # SCORE DE MOVIMENTO
    # ========================================================

    if len(agitation_scores) > 0:

        average_agitation = (
            sum(agitation_scores)
            / len(agitation_scores)
        )

    else:

        average_agitation = 0

    movement_score = min(
        average_agitation * 2,
        100
    )

    # ========================================================
    # SCORE DE POSTURA FECHADA
    # ========================================================

    closed_posture_score = (
        closed_posture_count / frame_count
    ) * 100

    # ========================================================
    # SCORE FINAL DE VÍDEO
    # ========================================================

    video_risk_score = (
        movement_score * 0.6
        + closed_posture_score * 0.4
    )

    video_risk_score = min(
        video_risk_score,
        100
    )

    # ========================================================
    # INTERPRETAÇÃO OBSERVÁVEL
    # ========================================================

    if movement_score >= 75:

        movement_pattern = "high movement"

    elif movement_score >= 50:

        movement_pattern = "moderate movement"

    else:

        movement_pattern = "low movement"

    if closed_posture_score >= 60:

        body_posture = "closed posture"

    else:

        body_posture = "open posture"

    return {

        "video_risk_score":
            round(video_risk_score, 2),

        "movement_pattern":
            movement_pattern,

        "body_posture":
            body_posture
    }


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_path: Path):

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.speech_recognition_language = "pt-BR"

    audio_config = speechsdk.audio.AudioConfig(
        filename=str(audio_path)
    )

    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    all_results = []

    speech_recognizer.recognized.connect(
        lambda evt: all_results.append(evt.result.text)
        if evt.result.text else None
    )

    speech_recognizer.start_continuous_recognition()

    time.sleep(10)

    speech_recognizer.stop_continuous_recognition()

    transcription = " ".join(all_results).strip()

    return transcription


# ============================================================
# ANÁLISE DE ÁUDIO
# ============================================================

def analyze_audio(transcription: str):

    language_client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    sentiment_response = language_client.analyze_sentiment(
        [transcription],
        language="pt"
    )

    sentiment_result = sentiment_response[0]

    key_phrase_response = language_client.extract_key_phrases(
        [transcription],
        language="pt"
    )

    key_phrases = key_phrase_response[0].key_phrases

    # --------------------------------------------------------
    # SCORE NEGATIVO
    # --------------------------------------------------------

    negative_score = (
        sentiment_result.confidence_scores.negative
    )

    # --------------------------------------------------------
    # FATOR DE PALAVRAS-CHAVE
    # --------------------------------------------------------

    keyword_factor = min(
        len(key_phrases) / 10,
        1.0
    )

    # --------------------------------------------------------
    # SCORE FINAL DE ÁUDIO
    # --------------------------------------------------------

    audio_risk_score = (
        (negative_score * 0.5)
        + (keyword_factor * 0.2)
    ) * 100

    return {

        "audio_risk_score":
            round(audio_risk_score, 2),

        "key_phrases":
            key_phrases[:5]
    }


# ============================================================
# FUSÃO MULTIMODAL
# ============================================================

def generate_final_result(
    transcription,
    video_result,
    audio_result
):

    final_risk_score = (
        video_result["video_risk_score"] * 0.5
        + audio_result["audio_risk_score"] * 0.5
    )

    if final_risk_score < 50:

        risk_level = "LOW"
        human_review = "human review required"

    elif final_risk_score < 75:

        risk_level = "MEDIUM"
        human_review = "human review optional"

    else:

        risk_level = "HIGH"
        human_review = "human review required"

    final_result = {

        "video_analysis": {

            "video_risk_score":
                video_result["video_risk_score"],

            "movement_pattern":
                video_result["movement_pattern"],

            "body_posture":
                video_result["body_posture"]
        },

        "audio_analysis": {

            "transcription":
                transcription,

            "audio_risk_score":
                audio_result["audio_risk_score"],

            "key_phrases":
                audio_result["key_phrases"]
        },

        "final_analysis": {

            "final_risk_score":
                round(final_risk_score, 2),

            "risk_level":
                risk_level,

            "human_review":
                human_review
        }
    }

    return final_result


# ============================================================
# EXECUÇÃO DO PIPELINE
# ============================================================

print("Executando análise de vídeo...\n")

video_result = analyze_video(
    VIDEO_PATH
)

print(
    json.dumps(
        {
            "video_analysis": video_result
        },
        indent=4,
        ensure_ascii=False
    )
)


print("\nNormalizando áudio...\n")

normalize_audio(
    AUDIO_PATH,
    AUDIO_PROCESSED_PATH
)


print("Executando Speech-to-Text...\n")

transcription = transcribe_audio(
    AUDIO_PROCESSED_PATH
)

print(
    json.dumps(
        {
            "transcription": transcription
        },
        indent=4,
        ensure_ascii=False
    )
)


print("\nExecutando análise de áudio...\n")

audio_result = analyze_audio(
    transcription
)

print(
    json.dumps(
        {
            "audio_analysis": audio_result
        },
        indent=4,
        ensure_ascii=False
    )
)


print("\nExecutando fusão multimodal...\n")

final_result = generate_final_result(
    transcription,
    video_result,
    audio_result
)


print("\nRESULTADO FINAL\n")

print(
    json.dumps(
        final_result,
        indent=4,
        ensure_ascii=False
    )
)


# ============================================================
# VISUALIZAÇÃO FINAL
# ============================================================

final_score = round(
    final_result["final_analysis"][
        "final_risk_score"
    ],
    2
)

risk_level = final_result[
    "final_analysis"
]["risk_level"]

human_review = final_result[
    "final_analysis"
]["human_review"]


# ------------------------------------------------------------
# DADOS DA TABELA
# ------------------------------------------------------------

table_data = [[
    final_score,
    risk_level,
    human_review
]]

column_labels = [
    "Final Score",
    "Risk Level",
    "Human Review"
]


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 2)
)

ax.axis("off")

table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="center"
)

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1, 2)


plt.title(
    "Final Analysis Summary",
    pad=20
)

plt.show()