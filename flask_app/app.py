from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
# from IndicTransToolkit import IndicProcessor
from gtts import gTTS
import soundfile as sf
from transformers import VitsTokenizer, VitsModel, set_seed
import os

# Initialize BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize IndicTrans model
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer_IT2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_IT2 = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
# ip = IndicProcessor(inference=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_IT2.to(DEVICE)

# MMS-TTS model for Marathi
mms_model_marathi = 'facebook/mms-tts-mar'

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, "image of", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def translate_caption(caption, target_languages):
    input_sentences = [caption]
    translations = {}
    for tgt_lang in target_languages:
        batch = ip.preprocess_batch(input_sentences, src_lang="eng_Latn", tgt_lang=tgt_lang)
        inputs = tokenizer_IT2(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_tokens = model_IT2.generate(**inputs, num_beams=5, max_length=256)
        translated_texts = tokenizer_IT2.batch_decode(generated_tokens, skip_special_tokens=True)
        translations[tgt_lang] = ip.postprocess_batch(translated_texts, lang=tgt_lang)[0]
    # return translations

    return {lang: f"Translation in {lang}" for lang in target_languages}

def generate_audio(text, lang_code, filename):
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    tts = gTTS(text=text, lang=lang_code)
    tts.save(audio_path)
    return audio_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle image upload
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file:
                # Save uploaded image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)

                # Generate caption
                caption = generate_caption(image_path)

                # Get selected languages
                target_languages = request.form.getlist("languages")
                translations = translate_caption(caption, target_languages)

                # Generate audio files for translations
                audio_files = {}
                lang_codes = {
                    "hin_Deva": "hi",  # Hindi
                    "mar_Deva": "mr",  # Marathi
                    "guj_Gujr": "gu",  # Gujarati
                    "urd_Arab": "ur"   # Urdu
                }

                for lang, translation in translations.items():
                    lang_code = lang_codes.get(lang, "en")  # Default to English
                    filename = f"{lang}.mp3"
                    audio_files[lang] = generate_audio(translation, lang_code, filename)

                return render_template(
                    "index.html",
                    uploaded_image=image_path,
                    caption=caption,
                    translations=translations,
                    audio_files=audio_files
                )
    return render_template("index.html")

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)