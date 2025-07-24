import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Reconnaissance de chiffre manuscrit", layout="centered")
st.title("🖌️ Dessinez un chiffre (0-9)")

# Zone de dessin
canvas_result = st_canvas(
    fill_color="black",  # Fond noir
    stroke_width=10,
    stroke_color="white",  # Couleur du trait
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Affiche l'image redimensionnée pour vérification
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    st.image(resized, caption="Image prétraitée (28x28)", width=150)

    # Convertir l'image en bytes
    im_pil = Image.fromarray(resized)
    buffered = io.BytesIO()
    im_pil.save(buffered, format="PNG")
    buffered.seek(0)

    if st.button("📤 Envoyer pour prédiction"):
        files = {"file": ("image.png", buffered, "image/png")}
        try:
            response = requests.post("http://localhost:8000/predict", files=files)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Chiffre prédit : {data['predicted_digit']} (confiance : {data['confidence']}%)")
                st.bar_chart(data["probabilities"])
            else:
                st.error(f"Erreur : {response.json().get('error')}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {str(e)}")