# synthesize_speech.py
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import torch
import os

def synthesize_speech(text, model, processor):
    inputs = processor(text, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    generated_speech = model.generate(inputs.input_ids)
    return generated_speech

def main(input_file, model_name):
    processor = Speech2TextProcessor.from_pretrained(model_name)
    model = Speech2TextForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    with open(input_file, "r") as f:
        captions = f.readlines()

    for line in captions:
        image, caption = line.strip().split(": ")
        audio = synthesize_speech(caption, model, processor)

        # Save the speech as an audio file
        audio_path = f"audio/{image.split('.')[0]}.wav"
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio.cpu().numpy())
        print(f"Generated speech for {image} saved at {audio_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize speech from captions using MMS TTS.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the file containing translated captions.")
    parser.add_argument("--model", type=str, required=True, help="Name of the TTS model (e.g., 'facebook/mms-tts').")
    args = parser.parse_args()

    main(args.input_file, args.model)
