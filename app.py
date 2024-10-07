import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from gtts import gTTS
import soundfile as sf
from transformers import VitsTokenizer, VitsModel, set_seed
import os

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer_IT2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_IT2 = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
ip = IndicProcessor(inference=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_IT2.to(DEVICE)

mms_models = {
    'hin_Deva': {'model_name': "facebook/mms-tts-hin", 'output_suffix': 'Hindi'},
    'mar_Deva': {'model_name': "facebook/mms-tts-mar", 'output_suffix': 'Marathi'},
    'guj_Gujr': {'model_name': "facebook/mms-tts-guj", 'output_suffix': 'Gujarati'}
}

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
    return translations

def generate_audio_gtts(text, lang_code, output_file):
    tts = gTTS(text=text, lang=lang_code)
    tts.save(output_file)
    return output_file

def generate_audio_fbmms(text, model_name, output_file):
    tokenizer = VitsTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    inputs = tokenizer(text=text, return_tensors="pt")
    set_seed(555)
    with torch.no_grad():
        outputs = model(**inputs)
    waveform = outputs.waveform[0].cpu().numpy()
    sf.write(output_file, waveform, samplerate=model.config.sampling_rate)
    return output_file

st.title("Multilingual Assistive Model")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Generating Caption...")
    caption = generate_caption(uploaded_image)
    st.write(f"Caption: {caption}")

    target_languages = st.multiselect(
        "Select target languages for translation", 
        ["hin_Deva", "mar_Deva", "guj_Gujr", "urd_Arab"],
        ["hin_Deva", "mar_Deva"]
    )

    if target_languages:
        st.write("Translating Caption...")
        translations = translate_caption(caption, target_languages)
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"{lang}: {translation}")
        
        for lang in target_languages:
            if lang in mms_models:
                tts_choice = st.radio(
                    f"Select TTS engine for {lang}:", 
                    options=["gTTS", "Facebook MMS-TTS"],
                    index=0
                )
                if tts_choice == "gTTS":
                    output_file = f"{lang}_gTTS.mp3"
                    lang_code = {
                        "hin_Deva": "hi",
                        "guj_Gujr": "gu",
                        "urd_Arab": "ur"
                    }.get(lang, "en")
                    audio_file = generate_audio_gtts(translations[lang], lang_code, output_file)
                else:
                    model_name = mms_models[lang]['model_name']
                    output_file = f"{lang}_MMS_TTS.{mms_models[lang]['output_suffix']}.wav"
                    audio_file = generate_audio_fbmms(translations[lang], model_name, output_file)
            else:
                st.write(f"Using gTTS for {lang}...")
                lang_code = {
                    "urd_Arab": "ur",
                }.get(lang, "en")
                output_file = f"{lang}_gTTS.mp3"
                audio_file = generate_audio_gtts(translations[lang], lang_code, output_file)

            st.write(f"Playing {lang} audio:")
            st.audio(audio_file)
else:
    st.write("Upload an image to start.")
