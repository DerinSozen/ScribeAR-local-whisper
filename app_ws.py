import asyncio
import websockets
import speech_recognition as sr
import threading
import numpy as np
import whisper
from scipy.signal import butter, lfilter
import numpy as np
from scipy.fftpack import fft

import json

from datetime import datetime, timedelta
from queue import Queue

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def contains_human_voice(audio_data, sample_rate, threshold=0.6):
    """
    Checks if the audio contains primarily human voice frequencies and prints detected frequencies.
    Args:
        audio_data: The audio data as a numpy array.
        sample_rate: The sample rate of the audio.
        threshold: The minimum ratio of energy in the human voice frequency band to total energy.
    Returns:
        True if the audio contains human voice frequencies, False otherwise.
    """
    # Perform FFT on the audio data
    audio_fft = fft(audio_data)
    audio_magnitude = np.abs(audio_fft)

    # Create frequency bins
    freqs = np.fft.fftfreq(len(audio_magnitude), 1/sample_rate)

    # Filter out negative frequencies (FFT produces symmetric result)
    freqs = freqs[:len(freqs) // 2]
    audio_magnitude = audio_magnitude[:len(audio_magnitude) // 2]

    # Calculate total energy in the frequency spectrum
    total_energy = np.sum(audio_magnitude)

    # Find and print the dominant frequencies (top 5) DEBUGGING 
    dominant_indices = np.argsort(audio_magnitude)[-5:]  # Get indices of top 5 values
    dominant_freqs = freqs[dominant_indices]  # Get corresponding frequencies
    dominant_amps = audio_magnitude[dominant_indices]  # Get corresponding amplitudes

    print("Dominant Frequencies (Hz):", dominant_freqs)
    print("Corresponding Amplitudes:", dominant_amps)

    # Define human voice frequency range (300 Hz to 3400 Hz)
    human_voice_indices = np.where((freqs >= 300) & (freqs <= 3400))[0]
    human_voice_energy = np.sum(audio_magnitude[human_voice_indices])

    # Calculate ratio of energy in human voice frequencies to total energy
    voice_energy_ratio = human_voice_energy / total_energy if total_energy > 0 else 0

    has_dominant_amp_above_8 = any(amp > 8 for amp in dominant_amps)

    # Check if the ratio exceeds the threshold
    return voice_energy_ratio > threshold and has_dominant_amp_above_8

lowcut = 300.0
highcut = 3400.0
fs = 16000  # Sample rate (already set to 16000 Hz)


# r = sr.Recognizer()

async def handler(websocket, path):
    
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    # use the default microphone as the audio source
    source = sr.Microphone(sample_rate=16000)
    # recorder.adjust_for_ambient_noise(source,  1)
    with sr.Microphone(sample_rate=16000) as source:                
        recorder.adjust_for_ambient_noise(source, 0.5)
        print("Done Calibrating.")
    
    # recorder.adjust_for_ambient_noise(source)
    # recorder.energy_threshold = 150
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # load model
    audio_model = whisper.load_model("tiny.en") 
    
    record_timeout = 2
    phrase_timeout = 3

    transcription = ['']
    

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)
    
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.")
    
    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                filtered_audio_np = bandpass_filter(audio_np, lowcut, highcut, fs)
                filtered_audio_np = filtered_audio_np.astype(np.float32)
                if (contains_human_voice(filtered_audio_np, fs)):

                # Time when transcription started
                    start_time = datetime.now()
                    # Read the transcription.
                    result = audio_model.transcribe(filtered_audio_np, fp16=False, temperature=0.0)
                    text = result['text'].strip()
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text
                    #Calculate the latency
                    latency = datetime.now() - start_time
                    # text = "Ftesting websocket"
                    await websocket.send('F'+transcription[-1]) 

                    #DEBUGGING INFO
                    print("-----------------------DEBUGGING------------------------")
                    print(len(transcription))
                    print("Transcribed Text:" + text)
                    print(f"Latency: {latency.total_seconds()} seconds")
                    print(recorder.energy_threshold)
                    print("-----------------------DEBUGGING------------------------")
                    #DEBUGGING INFO

        except KeyboardInterrupt:
            break


start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
