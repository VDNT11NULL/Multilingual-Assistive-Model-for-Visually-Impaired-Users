import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

@st.cache_resource
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

st.title("🧠 Multilingual Assistive VLM Prototype")

processor, model = load_blip2()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Generate Caption
    with st.spinner("Generating caption..."):
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        caption_ids = model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    st.success("🗨️ Caption:")
    st.markdown(f"> **{caption}**")

    question = st.text_input("Ask a question about the image (simulated STT):")

    if question:
        st.markdown(f"🧑‍💬 You asked: **{question}**")
        with st.spinner("Answering..."):
            qa_inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
            output_ids = model.generate(**qa_inputs, max_new_tokens=50)
            answer = processor.decode(output_ids[0], skip_special_tokens=True)

        st.success("🧠 Answer:")
        st.markdown(f"**{answer}**")
