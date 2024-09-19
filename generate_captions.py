# generate_captions.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

def generate_caption(image_path, model, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def main(input_dir):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    captions = {}
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        caption = generate_caption(image_path, model, processor)
        captions[image_file] = caption
        print(f"Caption for {image_file}: {caption}")

    # Save captions to a file
    with open("captions.txt", "w") as f:
        for img, caption in captions.items():
            f.write(f"{img}: {caption}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for images using BLIP.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    args = parser.parse_args()

    main(args.input_dir)
