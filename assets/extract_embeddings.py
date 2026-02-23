import onnxruntime
import numpy as np
import librosa
import argparse
import sys

SAMPLE_RATE = 24000
SPEECH_ENCODER_PATH = "./onnx32/speech_encoder.onnx"

def main():
    parser = argparse.ArgumentParser(description="Extract and save embeddings from audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--encoder", type=str, default=SPEECH_ENCODER_PATH, help="Path to speech encoder ONNX model")
    args = parser.parse_args()

    # Prepare audio input
    print(f"Loading audio from {args.audio}...")
    try:
        audio_values, _ = librosa.load(args.audio, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)
        
    audio_values = audio_values[np.newaxis, :].astype(np.float32)

    # Load ONNX session
    print(f"Loading Speech Encoder from {args.encoder}...")
    try:
        speech_encoder_session = onnxruntime.InferenceSession(args.encoder, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

    print("Running inference...")
    ort_speech_encoder_input = {"audio_values": audio_values}
    cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(None, ort_speech_encoder_input)
    
    # Save arrays to binary files
    cond_emb.tofile("cond_emb.bin")
    prompt_token.tofile("prompt_token.bin")
    speaker_embeddings.tofile("speaker_embeddings.bin")
    speaker_features.tofile("speaker_features.bin")
    
    # Save shapes to text file for reference
    with open("array_shapes.txt", "w") as f:
        f.write(f"cond_emb: {cond_emb.shape} dtype: {cond_emb.dtype}\n")
        f.write(f"prompt_token: {prompt_token.shape} dtype: {prompt_token.dtype}\n")
        f.write(f"speaker_embeddings: {speaker_embeddings.shape} dtype: {speaker_embeddings.dtype}\n")
        f.write(f"speaker_features: {speaker_features.shape} dtype: {speaker_features.dtype}\n")
    
    print(f"Saved arrays to binary files:")
    print(f"  cond_emb.bin - shape: {cond_emb.shape}, dtype: {cond_emb.dtype}")
    print(f"  prompt_token.bin - shape: {prompt_token.shape}, dtype: {prompt_token.dtype}")
    print(f"  speaker_embeddings.bin - shape: {speaker_embeddings.shape}, dtype: {speaker_embeddings.dtype}")
    print(f"  speaker_features.bin - shape: {speaker_features.shape}, dtype: {speaker_features.dtype}")
    print("Done!")

if __name__ == "__main__":
    main()
