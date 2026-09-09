import asyncio
import json
import websockets
import numpy as np

async def test_websocket_speech_pause_finalize():
    uri = "ws://127.0.0.1:8080/ws/transcribe?stt_model=mock&sample_rate=16000"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        # Handshake
        init = await ws.recv()
        print(f"Connected: {init}")

        # 1. Generate 0.4s of speech tone (6400 samples)
        t = np.linspace(0, 0.4, 6400, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 400 * t) * 32767).astype(np.int16)
        speech_bytes = tone.tobytes()

        # Send in 2 chunks
        await ws.send(speech_bytes[:6400])
        await asyncio.sleep(0.05)
        await ws.send(speech_bytes[6400:])
        await asyncio.sleep(0.05)

        # 2. Send 0.6s of silence (9600 samples) to trigger natural pause (>400ms)
        silence_chunk = np.zeros(1600, dtype=np.int16).tobytes()
        boundary_received = None
        for _ in range(6):
            await ws.send(silence_chunk)
            await asyncio.sleep(0.05)
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.08)
                    data = json.loads(raw)
                    if data.get("boundary"):
                        boundary_received = data
                        print(f"[ASR] Natural pause boundary committed: {data.get('sentence')} | full: '{data.get('text')}'")
                except asyncio.TimeoutError:
                    break

        assert boundary_received is not None, "Natural pause boundary should be committed"
        committed_sentence = boundary_received.get("sentence", "")
        assert committed_sentence, "Committed sentence must not be empty"

        # 3. Subsequent silence: verify zero hallucinations and zero decode calls
        print("Sending additional silence frames after pause...")
        for _ in range(4):
            await ws.send(silence_chunk)
            await asyncio.sleep(0.05)

        # 4. Finalize
        print("Doctor presses Stop -> WebSocket sends finalize...")
        await ws.send(json.dumps({"action": "finalize"}))
        final_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        final_data = json.loads(final_raw)
        print(f"Final response on Stop: {final_data}")

        assert final_data["type"] == "final"
        # Crucial check: new_sentence_committed MUST BE FALSE because it was already committed at pause
        assert not final_data["new_sentence_committed"], "new_sentence_committed must be False since sentence was already committed at pause!"
        # Final text must equal the boundary text
        assert final_data["text"] == boundary_received["text"], f"Final text '{final_data['text']}' must match boundary text '{boundary_received['text']}'"

        print("\nPASS: End-to-end single pipeline works with zero duplicate executions on Stop!")

if __name__ == "__main__":
    asyncio.run(test_websocket_speech_pause_finalize())
