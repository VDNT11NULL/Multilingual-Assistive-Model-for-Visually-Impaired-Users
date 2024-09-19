# translate_captions.py
from indictrans import Transliterator

def translate_captions(captions_file, lang):
    with open(captions_file, "r") as f:
        captions = f.readlines()

    trn = Transliterator(source='eng', target=lang)
    translations = {}
    for line in captions:
        image, caption = line.strip().split(": ")
        translated_caption = trn.transform(caption)
        translations[image] = translated_caption
        print(f"Translated {image}: {translated_caption}")

    # Save translated captions
    with open(f"translated_captions_{lang}.txt", "w") as f:
        for img, trans_caption in translations.items():
            f.write(f"{img}: {trans_caption}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Translate captions into Indic languages using IndicTrans2.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the file containing captions.")
    parser.add_argument("--lang", type=str, required=True, help="Target language code (e.g., 'hi' for Hindi, 'mr' for Marathi).")
    args = parser.parse_args()

    translate_captions(args.input_file, args.lang)
