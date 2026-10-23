import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import os
import io

st.set_page_config(
    page_title="Face Detection AI",
    page_icon="🧑",
    layout="centered"
)

MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/"
    "opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_face_detector():
    # Download the model if it is missing or empty.
    if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()

        if b"<opencv_storage>" not in response.content:
            raise RuntimeError("Downloaded face model is invalid.")

        with open(MODEL_PATH, "wb") as model_file:
            model_file.write(response.content)

    detector = cv2.CascadeClassifier(MODEL_PATH)

    if detector.empty():
        raise RuntimeError("OpenCV could not load the face model.")

    return detector


st.title("🧑 Face Detection System AI")
st.write("Upload a photograph to detect faces.")

st.info(
    "Privacy: Uploaded images are processed by this application. "
    "When hosted online, images are sent to the hosting server."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        st.subheader("Original Image")
        st.image(image_array, use_container_width=True)

        if st.button("🔍 Detect Faces"):
            with st.spinner("Loading model and detecting faces..."):
                detector = load_face_detector()

                gray = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2GRAY
                )

                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                result = image_array.copy()

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        result,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

            st.success(f"✅ Faces Detected: {len(faces)}")

            st.subheader("Detection Result")
            st.image(
                result,
                caption="Faces outlined in green",
                use_container_width=True
            )

            output = io.BytesIO()
            Image.fromarray(result).save(output, format="PNG")

            st.download_button(
                "⬇️ Download Result",
                data=output.getvalue(),
                file_name="face_detection_result.png",
                mime="image/png"
            )

    except Exception as error:
        st.error("Face detection failed.")
        st.exception(error)

else:
    st.warning("Upload an image to get started.")
