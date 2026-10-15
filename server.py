import urllib.parse
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pipeline import MultilingualS2STPipeline
import uvicorn

app = FastAPI(title="Multilingual S2ST Hardware Bridge")
pipeline = MultilingualS2STPipeline()
pipeline.warmup("en", "hin_Deva", "hin")

# Endpoint 1: HTTP Push-to-Talk (Synchronous def so it won't block the server loop)
@app.post("/translate-audio")
def translate_audio_endpoint(
    audio_file: UploadFile = File(...),
    target_lang: str = Form("hin_Deva"),
    tts_code: str = Form("hin")
):
    # Read uploaded audio bytes
    input_bytes = audio_file.file.read()
    
    # Run the ML pipeline
    result = pipeline.run_pipeline(input_bytes, target_lang, tts_code)
    
    # URL-encode non-ASCII strings for safe HTTP header transport (fixes latin-1 crash)
    safe_transcript = urllib.parse.quote(result["transcript"])
    safe_translation = urllib.parse.quote(result["translation"])

    return Response(
        content=result["audio_bytes"],
        media_type="audio/wav",
        headers={
            "X-Transcript": safe_transcript,
            "X-Translation": safe_translation
        }
    )

# Endpoint 2: WebSocket for hardware stream
@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    try:
        while True:
            # Receive audio payload from hardware
            data = await websocket.receive_bytes()
            if not data:
                continue
            
            # Default target: Hindi (or make dynamic via query params/json payload)
            import asyncio
            result = await asyncio.to_thread(pipeline.run_pipeline, data, "hin_Deva", "hin")
            
            # Send back synthesized audio bytes
            await websocket.send_bytes(result["audio_bytes"])
    except WebSocketDisconnect:
        print("Hardware client disconnected cleanly.")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)