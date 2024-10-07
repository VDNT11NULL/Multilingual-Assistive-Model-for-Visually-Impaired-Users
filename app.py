import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
import os
from IndicTransToolkit import IndicProcessor
from gtts import gTTS
import soundfile as sf
from transformers import VitsTokenizer, VitsModel, set_seed

# Clone and Install IndicTransToolkit repository
if not os.path.exists('IndicTransToolkit'):
    os.system('git clone https://github.com/VarunGumma/IndicTransToolkit')
    os.system('cd IndicTransToolkit && python3 -m pip install --editable ./')

# Ensure that IndicTransToolkit is installed and used properly
from IndicTransToolkit import IndicProcessor

# Initialize BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate captions
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, "image of", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Function for translation using IndicTrans2
def translate_caption(caption, target_languages):
    # Load model and tokenizer
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer_IT2 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_IT2 = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model_IT2 = torch.quantization.quantize_dynamic(
        model_IT2, {torch.nn.Linear}, dtype=torch.qint8
    )

    ip = IndicProcessor(inference=True)

    # Source language (English)
    src_lang = "eng_Latn"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_IT2.to(DEVICE)  # Move model to the device

    # Integrating with workflow now
    input_sentences = [caption]
    translations = {}

    for tgt_lang in target_languages:
        # Preprocess input sentences
        batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the sentences and generate input encodings
        inputs = tokenizer_IT2(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model_IT2.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer_IT2.as_target_tokenizer():
            generated_tokens = tokenizer_IT2.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Postprocess the translations
        translated_texts = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        translations[tgt_lang] = translated_texts[0]

    return translations

# Function to generate audio using gTTS
def generate_audio_gtts(text, lang_code, output_file):
    tts = gTTS(text=text, lang=lang_code)
    tts.save(output_file)
    return output_file

# Function to generate audio using Facebook MMS-TTS
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
        ["hin_Deva", "mar_Deva", "guj_Gujr", "urd_Arab"],  # Add more languages as needed
        ["hin_Deva", "mar_Deva"]
    )

    # Generate Translations
    if target_languages:
        st.write("Translating Caption...")
        translations = translate_caption(caption, target_languages)
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"{lang}: {translation}")
        
        # Default to gTTS for TTS
        for lang in target_languages:
            st.write(f"Using gTTS for {lang}...")
            lang_code = {
                "hin_Deva": "hi",  # Hindi
                "guj_Gujr": "gu",  # Gujarati
                "urd_Arab": "ur"   # Urdu
            }.get(lang, "en")
            output_file = f"{lang}_gTTS.mp3"
            audio_file = generate_audio_gtts(translations[lang], lang_code, output_file)

            st.write(f"Playing {lang} audio:")
            st.audio(audio_file)
else:
    st.write("Upload an image to start.")
