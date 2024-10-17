# test_whisper_model.py

import whisper
import pytest

@pytest.fixture
def audio_model():
    return whisper.load_model("base.en")

def test_whisper_model_loaded(audio_model):
    assert audio_model is not None

def test_whisper_transcribe(audio_model):
    audio_np = ... # Provide sample audio data
    result = audio_model.transcribe(audio_np, fp16=False, temperature=0.0)
    assert isinstance(result, dict)
    assert 'text' in result