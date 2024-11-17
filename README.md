# Multilingual-Assistive-Model-for-Visually-Impaired-Users

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/VDNT11/Multilingual-Assistive-Model)

---

This repository provides a working demonstration of a Multilingual Assistive Model designed to offer support for visually impaired users by integrating image captioning, translation, and text-to-speech (TTS) synthesis. By generating descriptive and multilingual content in audio format, the system aims to provide a seamless user experience across multiple Indic languages.

------

## **Table of Contents**
1. [Abstract](#abstract)
2. [Features](#features)
3. [Methodology](#methodology)
4. [Model Inference](#model-inference)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Improvements](#future-improvements)
9. [Acknowledgments](#Acknowledgments)

---

## **Abstract**

This project focuses on developing a comprehensive assistive system that combines state-of-the-art models to generate meaningful and accessible audio descriptions for visually impaired users. The workflow integrates **image captioning** using the **BLIP** model, **translation** via **IndicTrans2**, and **speech synthesis** utilizing both **gTTS** and **Facebook's Vits MMS-TTS model**.

The model performs the following tasks:
- Image captioning using BLIP to generate descriptions of visual content.
- Translation of the captions into multiple Indic languages using IndicTrans2.
- Text-to-speech conversion using **Facebook’s MMS-TTS** for multilingual support across 1000+ languages.

---

## **Features**
- **Image Captioning**: Describes the content of images using natural language.
- **Multilingual Support**: Translates captions into multiple Indian languages (Hindi, Marathi, Gujarati, etc.).
- **Text-to-Speech (TTS)**: Provides audio outputs for visually impaired users using TTS models, including **Facebook's Vits MMS-TTS**.
- **Fully Offline**: Reduces dependency on external APIs, leveraging pre-trained models for local inference.

---

## **Methodology**

The project follows a structured approach to integrate multiple machine learning models:

### **1. Model Selection**:
- **Image Captioning**: 
  - We use **BLIP (Bootstrapped Language Image Pretraining)**, a model by Salesforce, for generating captions from images. It is pre-trained on the MS COCO dataset and is known for its superior performance in image captioning tasks.
  
- **Translation**:
  - The **IndicTrans2** model is employed to translate the captions into Indic languages, offering support for languages such as Hindi, Marathi, and Gujarati.
  
- **Speech Synthesis**:
  - For speech synthesis, we use two models:
    - **gTTS (Google Text-to-Speech)** for basic TTS.
    - **Facebook’s Vits MMS model** for synthesizing speech in multiple languages, including low-resource Indic languages.

### **2. Inference Pipeline**:
- **Generating Captions**: The BLIP model generates descriptive captions from input images.
- **Translation**: The captions are translated into selected Indian languages using IndicTrans2.
- **Speech Synthesis**: The translated captions are then converted to audio using Facebook's MMS-TTS model to ensure high-quality speech output.

---

## **Model Inference**

### **Facebook's MMS-TTS Inference**:
- Facebook's **Vits MMS model** supports TTS synthesis in over 1000 languages, making it highly suitable for diverse and low-resource languages like Hindi. While APIs like Google Translate are available for TTS, this model provides offline capabilities and better performance for certain regional languages.
  
- **Image Captioning** using BLIP provides detailed descriptions, which are translated using **IndicTrans2**, and speech is synthesized with the **MMS-TTS** model, providing an end-to-end multilingual solution.

---

## **Installation**

To install the required dependencies, clone the repository and install the necessary libraries:

```bash
git clone https://github.com/your-repo/multilingual-assistive-model.git
cd multilingual-assistive-model
pip install -r requirements.txt
```

### **Dependencies**
- `torch`
- `transformers`
- `gtts`
- `indictrans`
- `mms-tts`

---

## **Usage**

To run the model on a set of input images, execute the following commands:

1. **Generate Captions**:
   ```bash
   python generate_captions.py --input_dir path_to_images
   ```

2. **Translate Captions**:
   ```bash
   python translate_captions.py --input_file captions.txt --lang hi
   ```

3. **Synthesize Speech**:
   ```bash
   python synthesize_speech.py --input_file translated_captions.txt --model mms_tts
   ```

---

## **Results**

After running the pipeline, the model will generate:
- **Captions**: Descriptions for each input image.
- **Translated Captions**: Captions translated into the desired Indic language.
- **Audio Files**: Audio narration for each translated caption, synthesized using the MMS-TTS model.

---

## **Future Improvements**

- Extend support for more regional languages, improving translation accuracy for lower-resource languages.
- Deployment of the project on open-source platforms using streamlit.
- Explore additional state-of-the-art models for speech synthesis that perform well with complex grammatical structures in Indic languages.


---

### **Acknowledgments**

- **Salesforce** for the **BLIP** model.
- **Facebook AI** for the **MMS-TTS** model.
- **AI4Bharat** for the **IndicTrans2** model.

---
