import numpy as np
import pytest

def test_audio_data_conversion():
    audio_data = b'\x00\x01\x02\x03'
    expected_audio_np = np.array([0.0, 0.00003052, 0.00006104, 0.00009155], dtype=np.float32)
    
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    assert np.allclose(audio_np, expected_audio_np)

def test_audio_data_empty():
    audio_data = b''
    expected_audio_np = np.array([], dtype=np.float32)
    
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    assert np.array_equal(audio_np, expected_audio_np)