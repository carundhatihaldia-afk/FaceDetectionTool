import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Face Detection AI",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Face Detection System AI")
st.write("Upload an image to detect faces automatically.")

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
            gray = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2GRAY
            )

            cascade_path = os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml"
            )

            if not os.path.isfile(cascade_path):
                st.error(
                    "Face detection model file was not found."
                )
                st.code(cascade_path)

            else:
                face_detector = cv2.CascadeClassifier(
                    cascade_path
                )

                if face_detector.empty():
                    st.error(
                        "OpenCV could not load the face detection model."
                    )
                else:
                    faces = face_detector.detectMultiScale(
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

                    st.success(
                        f"✅ Faces Detected: {len(faces)}"
                    )

                    st.subheader("Detection Result")
                    st.image(
                        result,
                        caption="Detected faces highlighted in green",
                        use_container_width=True
                    )

                    output = io.BytesIO()
                    Image.fromarray(result).save(
                        output,
                        format="PNG"
                    )

                    st.download_button(
                        label="⬇️ Download Result",
                        data=output.getvalue(),
                        file_name="face_detection_result.png",
                        mime="image/png"
                    )

    except Exception as e:
        st.error("An error occurred while processing the image.")
        st.exception(e)

else:
    st.info("Upload an image to get started.")
