import asyncio
import websockets
import speech_recognition as sr
import threading
import numpy as np
import speech_recognition as sr
import whisper
import re
import json
import time

from datetime import datetime, timedelta
from queue import Queue

r = sr.Recognizer()

async def handler(websocket, path):
    for i in range(100):
        time.sleep(1)
        text = "Ftesting websocket"
        await websocket.send((text)) 


start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()