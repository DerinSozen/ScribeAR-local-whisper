import speech_recognition as sr
import pytest

@pytest.fixture
def recognizer():
    return sr.Recognizer()

def test_recognizer_energy_threshold(recognizer):
    assert recognizer.energy_threshold == 1000

def test_recognizer_dynamic_energy_threshold(recognizer):
    assert recognizer.dynamic_energy_threshold == False