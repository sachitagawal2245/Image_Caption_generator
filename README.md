# 🖼️ Image Caption Generator using CNN + LSTM

A deep learning model that automatically generates meaningful captions for images by combining **ResNet50** for visual feature extraction and **LSTM** for natural language generation.

---

## 🚀 Demo

> Upload an image → Get a caption instantly!

.

---

##  How It Works

```
Image Input → ResNet50 (Feature Extraction) → LSTM (Caption Generation) → Caption Output
```

1. **ResNet50 (CNN)** extracts a feature vector from the input image
2. The feature vector is fed into an **LSTM** model along with tokenized caption sequences
3. The LSTM generates captions **word-by-word** until the end token is reached

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, Pandas |
| Deployment | Streamlit |
| Model | ResNet50 + LSTM |

---

```

---

##  Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/sachitagawal2245/Image_Caption_generator.git
cd Image_Caption_generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---


```

- Dataset used: **Flickr8k**
- Captions are cleaned, tokenized, and padded before training
- A custom data generator is used for efficient batch loading

---

> Currently working on improving accuracy using **attention mechanisms** and **transformer-based models**.

---

##  Future Improvements

- [ ] Add **Attention Mechanism** for better caption quality
- [ ] Experiment with **Transformer-based** models (e.g. ViT + GPT-2)
- [ ] Evaluate using **BLEU scores**
- [ ] Train on larger datasets (Flickr30k, MS-COCO)

---

##  Author

**Sachit Agarwal**  
📧 sachitagarwal244@gmail.com  
🔗 [LinkedIn]www.linkedin.com/in/sachit-agarwal-8282461b7 | [GitHub](https://github.com/sachitagawal2245)

---

##  Show Your Support

If you found this project helpful, please consider giving it a **star** ⭐ on GitHub!
