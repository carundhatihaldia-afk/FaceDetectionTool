import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import io
import os

st.set_page_config(
    page_title="Face Detection System AI",
    page_icon="🧑",
    layout="centered"
)

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/"
    "opencv/master/samples/dnn/face_detector/deploy.prototxt"
)

MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/"
    "opencv_3rdparty/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

PROTOTXT_PATH = "/tmp/deploy.prototxt"
MODEL_PATH = "/tmp/face_detector.caffemodel"

st.title("🧑 Face Detection System AI")
st.write("Upload a photograph to detect faces and highlight them with rectangles.")

st.info(
    "🔒 Privacy: Images uploaded to this hosted application "
    "are sent to its server for processing."
)

with st.expander("OpenCV Diagnostics"):
    st.write("OpenCV version:", getattr(cv2, "__version__", "Unknown"))
    st.write("OpenCV location:", getattr(cv2, "__file__", "Unknown"))
    st.write("DNN module available:", hasattr(cv2, "dnn"))

    if hasattr(cv2, "dnn"):
        st.write(
            "Caffe loader available:",
            hasattr(cv2.dnn, "readNetFromCaffe")
        )


@st.cache_resource
def load_face_detector():
    if not hasattr(cv2, "dnn"):
        raise RuntimeError("OpenCV DNN is unavailable.")

    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise RuntimeError(
            "cv2.dnn.readNetFromCaffe is missing. "
            "Check OpenCV Diagnostics and requirements.txt."
        )

    if not os.path.isfile(PROTOTXT_PATH) or os.path.getsize(PROTOTXT_PATH) == 0:
        response = requests.get(PROTOTXT_URL, timeout=60)
        response.raise_for_status()

        with open(PROTOTXT_PATH, "wb") as file:
            file.write(response.content)

    if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        response = requests.get(MODEL_URL, timeout=180)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)

    if os.path.getsize(PROTOTXT_PATH) < 1000:
        raise RuntimeError("The downloaded model configuration is invalid.")

    if os.path.getsize(MODEL_PATH) < 1000000:
        raise RuntimeError("The downloaded face-detection model is incomplete.")

    return cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
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

        if st.button("🔍 Detect Faces", type="primary"):
            with st.spinner("Detecting faces..."):
                detector = load_face_detector()

                height, width = image_array.shape[:2]

                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image_array, (300, 300)),
                    scalefactor=1.0,
                    size=(300, 300),
                    mean=(104.0, 177.0, 123.0)
                )

                detector.setInput(blob)
                detections = detector.forward()

                result = image_array.copy()
                face_count = 0

                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])

                    if confidence < 0.50:
                        continue

                    box = (
                        detections[0, 0, i, 3:7]
                        * np.array([width, height, width, height])
                    )

                    x1, y1, x2, y2 = box.astype("int")

                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_count += 1

                    cv2.rectangle(
                        result,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        result,
                        f"Face {face_count}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            if face_count:
                st.success(f"✅ Faces Detected: {face_count}")
            else:
                st.warning(
                    "No faces detected. Try a clear, front-facing photograph."
                )

            st.subheader("Detection Result")
            st.image(
                result,
                caption="Detected faces outlined in green",
                use_container_width=True
            )

            output = io.BytesIO()
            Image.fromarray(result).save(output, format="PNG")

            st.download_button(
                label="⬇️ Download Result",
                data=output.getvalue(),
                file_name="face_detection_result.png",
                mime="image/png"
            )

    except requests.RequestException as error:
        st.error("Could not download the face-detection model.")
        st.code(str(error))

    except Exception as error:
        st.error("Face detection failed. Diagnostic details:")
        st.exception(error)

else:
    st.warning("Upload a JPG, JPEG, or PNG photograph to get started.")

st.divider()
st.caption("Face Detection AI | Python • Streamlit • OpenCV DNN")
