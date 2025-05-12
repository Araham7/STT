import os
import wave
import json
import subprocess
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer

app = Flask(__name__)

# Load model at startup
try:
    MODEL = Model("./model")  # Consider making this path configurable
except Exception as e:
    app.logger.error(f"Failed to load Vosk model: {str(e)}")
    raise

def convert_audio(input_path: str, output_path: str) -> bool:
    """Convert audio to 16KHz mono WAV format using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Audio conversion failed: {str(e)}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error during audio conversion: {str(e)}")
        return False

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Vosk model."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise ValueError("Audio file must be WAV format mono PCM.")

            recognizer = KaldiRecognizer(MODEL, wf.getframerate())
            recognizer.SetWords(True)  # Enable word-level timestamps if needed

            result = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result.append(json.loads(recognizer.Result()))

            # Get final result
            result.append(json.loads(recognizer.FinalResult()))
            
            # Combine all partial results
            return ' '.join([res.get('text', '') for res in result if res.get('text')])
            
    except Exception as e:
        app.logger.error(f"Transcription failed: {str(e)}")
        raise

@app.route('/stt', methods=['POST'])
def speech_to_text():
    """Endpoint for speech-to-text conversion."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Save uploaded file
        audio_file = request.files['file']
        temp_path = 'temp_upload.wav'
        converted_path = 'converted.wav'
        audio_file.save(temp_path)

        # Convert audio format
        if not convert_audio(temp_path, converted_path):
            return jsonify({'error': 'Audio conversion failed'}), 400

        # Transcribe audio
        transcription = transcribe_audio(converted_path)
        
        return jsonify({
            'text': transcription,
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"STT processing error: {str(e)}")
        return jsonify({'error': 'Speech recognition failed', 'details': str(e)}), 500

    finally:
        # Cleanup temporary files
        for file_path in [temp_path, converted_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                app.logger.warning(f"Could not delete temp file {file_path}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



    # curl -X POST http://localhost:5000/stt   -F "file=@04.ogg"