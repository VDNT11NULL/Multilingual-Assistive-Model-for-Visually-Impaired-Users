import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
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
ip = IndicProcessor(inference=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_IT2.to(DEVICE)

# MMS-TTS model for Marathi
mms_model_marathi = 'facebook/mms-tts-mar'

# Function to generate captions
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, "image of", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Function for translation using IndicTrans2
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
    return translations

# Function to generate audio using gTTS
def generate_audio_gtts(text, lang_code, output_file):
    tts = gTTS(text=text, lang=lang_code)
    tts.save(output_file)
    return output_file

# Function to generate audio using Facebook MMS-TTS for Marathi
def generate_audio_fbmms(text, output_file):
    tokenizer = VitsTokenizer.from_pretrained(mms_model_marathi)
    model = VitsModel.from_pretrained(mms_model_marathi)
    inputs = tokenizer(text=text, return_tensors="pt")
    set_seed(555)
    with torch.no_grad():
        outputs = model(**inputs)
    waveform = outputs.waveform[0].cpu().numpy()
    sf.write(output_file, waveform, samplerate=model.config.sampling_rate)
    return output_file

# Streamlit UI
st.title("Multilingual Assistive Model")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate Caption
    st.write("Generating Caption...")
    caption = generate_caption(uploaded_image)
    st.write(f"Caption: {caption}")

    # Select target languages for translation
    target_languages = st.multiselect(
        "Select target languages for translation", 
        ["hin_Deva", "mar_Deva", "guj_Gujr"],
        ["hin_Deva", "mar_Deva"]
    )

    # Generate Translations
    if target_languages:
        st.write("Translating Caption...")
        translations = translate_caption(caption, target_languages)
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"{lang}: {translation}")
        
        # Choice for TTS for Marathi (gTTS vs MMS-TTS)
        if "mar_Deva" in translations:
            marathi_tts_choice = st.radio(
                "Select TTS engine for Marathi",
                options=["gTTS", "Facebook MMS-TTS"],
                index=0  # Default to gTTS
            )
        
        # Generate TTS Audio
        audio_files = {}
        for lang, translation in translations.items():
            if lang == "mar_Deva":
                if marathi_tts_choice == "gTTS":
                    st.write(f"Using gTTS for Marathi...")
                    output_file = f"{lang}_gTTS.mp3"
                    audio_file = generate_audio_gtts(translation, "mr", output_file)
                else:
                    st.write(f"Using Facebook MMS-TTS for Marathi...")
                    output_file = f"{lang}_MMS_TTS.wav"
                    audio_file = generate_audio_fbmms(translation, output_file)
            else:
                st.write(f"Using gTTS for {lang}...")
                lang_code = {
                    "hin_Deva": "hi",  # Hindi
                    "guj_Gujr": "gu",  # Gujarati
                }.get(lang, "en")
                output_file = f"{lang}_gTTS.mp3"
                audio_file = generate_audio_gtts(translation, lang_code, output_file)

            audio_files[lang] = audio_file
            st.write(f"Playing {lang} audio:")
            st.audio(audio_file)
else:
    st.write("Upload an image to start.")

