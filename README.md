# ⚡️ Smart Split Image Compressor

✨ Cutting-edge side-by-side image compression lab: Classic, Modern, and AI Native Algorithms in One Interactive Web App!

🔗 https://smart-split-image-compress.streamlit.app/
---

## 🚀 What is this app?

This Streamlit web app lets you **upload any image and experience the magic of image compression with a twist**:

- Protect important ("ROI") areas by drawing rectangles—see how the app preserves faces or text at higher quality.
- Explore and compare a rich suite of algorithms: from classic SVD, JPEG, and DCT, to mathy PCA and KMeans, modern wavelet and and even neural-network-powered autoencoder compression.
- Instantly see where your image stays sharp and where the smart compressor saves space.
- Visualize adaptive compression with overlays and heatmaps.
- Download your compressed result!

---

## 🧑‍💻 Features

- **🖼️ Easy Upload:** PNG, JPG/JPEG—classic and modern images supported.
- **Draw or Detect:** Mark regions to keep at higher quality, or let the app auto-detect faces.
- **Classic Compression:** SVD (Singular Value Decomposition), DCT (Discrete Cosine Transform), JPEG (PIL), and PCA.
- **Stylized/AI-inspired:** KMeans (cartoon-style), Wavelet, and Autoencoder neural compression.
- **Adaptive:** Compression level changes for regions you care about most.
- **Interactive overlay:** Visual feedback showing preserved/sharply-compressed areas vs. highly-compressed.
- **Heatmap Quality Map:** Instantly see tilewise compression strengths.
- **Remove selected regions:** Unmark with a click!
- **Download compressed output** with a single button.
- **Modern, educational, and fun:** Great for learning compression, demonstrating AI, and sharing visual results.

---

## 🎨 Algorithms Overview

| Algorithm     | What it Does                                           | Visual Style                      |
|---------------|-------------------------------------------------------|-----------------------------------|
| SVD           | Matrix factorization for low-rank approximation       | Soft, watercolor, abstract        |
| DCT           | Frequency transform (core of JPEG)                    | Classic blocks, smooth/blurry     |
| PIL (JPEG)    | Standard JPEG encoder                                 | Standard, fast, universally used  |
| PCA           | Keeps most "important" variance                       | Minimal detail, mathy "core"      |
| KMeans        | Palette-reduced vector quantization                   | Cartoonized, bold color patches   |
| Wavelet       | Multi-scale transform, block-free                     | Smooth, fewer hard artifacts      |
| Autoencoder   | Neural network (AI) compression                       | Dreamy, neural, new-gen           |

---

## 🛠️ How to Use

1. **Upload your image** (PNG, JPG).
2. **Mark important areas**:
    - Draw rectangles on faces/text, or enable auto-detect faces.
3. **Choose a compression algorithm** from the dropdown.
4. **Adjust compression level** and toggle overlays/heatmaps.
5. **Download the compressed image** with a click.

---

## 🖥️ Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/ayush-git228/smart-split-image-compressor.git
    cd smart-split-image-compressor
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Run the app:**
    ```
    streamlit run main.py
    ```

---

## 🤩 Why is this app fascinating?

- **Live interactive comparison**: See traditional vs. AI compression side-by-side.
- **Protect key image regions** with a simple drawing tool.
- **Visual heatmaps** expose where the app spends quality "budget".

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss any major changes.
Can add any Algorithm or method which can make this app more fascinating or useful.
---


## 📝 Requirement

At present, I am unable to enable the draw/brush tool on the image, only the rectangle tool works as expected. 
If you know how to activate and use the freehand drawing or brush mode alongside rectangles in streamlit-drawable-canvas, guidance would be appreciated.
---


Based on Streamlit, OpenCV, Pillow, NumPy, scikit-learn, and other open-source projects.


