# test_websocket_handler.py

import asyncio
import websockets
import pytest

@pytest.mark.asyncio
async def test_websocket_handler():
    async with websockets.connect("ws://localhost:1234") as websocket:
        # Send sample audio data to the websocket
        await websocket.send(b'\x00\x01\x02\x03')
        
        # Wait for the response
        response = await websocket.recv()
        
        # Assert the response
        assert isinstance(response, str)
        # Add more specific assertions based on expected behavior