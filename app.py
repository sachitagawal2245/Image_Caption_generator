import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ------------------ LOAD All models ------------------

@st.cache_resource
def load_all():
    model = load_model("best_caption_model.keras", compile=False)
    with open("data/processed/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("data/processed/idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)
    with open("data/processed/max_length.pkl", "rb") as f:
        max_length = pickle.load(f)
    extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model, word2idx, idx2word, max_length, extractor


# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# ------------------ CAPTION GENERATOR ------------------
def generate_caption(model, feature, word2idx, idx2word, max_length):
    text = "startseq"
    for _ in range(max_length):
        seq = [word2idx.get(w, word2idx["<unk>"]) for w in text.split()]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = model.predict([feature, seq], verbose=0)
        pred = np.argmax(pred)
        word = idx2word.get(pred)
        if word is None or word == "<unk>":
            break
        text += " " + word
        if word == "endseq":
            break
    return text.replace("startseq", "").replace("endseq", "").strip()


# ------------------ UI ------------------

st.title("🖼️ Image Caption Generator")
model,word2idx,idx2word, max_length,extractor=load_all()
uploaded_file=st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image=Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            img = preprocess_image(image)
            feature = extractor.predict(img, verbose=0)
            caption = generate_caption(
                model, feature, word2idx, idx2word, max_length
            )
        st.success(caption)