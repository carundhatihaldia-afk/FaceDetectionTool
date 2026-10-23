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

CONFIDENCE_THRESHOLD = 0.75
MIN_FACE_SIZE = 30

st.title("🧑 Face Detection System AI")

st.write(
    "Upload a photograph to detect faces and highlight them "
    "with green rectangles."
)

st.info(
    "🔒 Privacy: Uploaded images are processed on the server "
    "hosting this application."
)

with st.expander("OpenCV Diagnostics"):
    st.write("OpenCV version:", cv2.__version__)
    st.write("OpenCV location:", cv2.__file__)
    st.write("DNN module available:", hasattr(cv2, "dnn"))
    st.write(
        "Caffe loader available:",
        hasattr(cv2.dnn, "readNetFromCaffe")
        if hasattr(cv2, "dnn")
        else False
    )


@st.cache_resource
def load_face_detector():
    if not hasattr(cv2, "dnn"):
        raise RuntimeError("OpenCV DNN is unavailable.")

    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        raise RuntimeError(
            "OpenCV does not provide readNetFromCaffe. "
            "Check your installed OpenCV package."
        )

    if (
        not os.path.isfile(PROTOTXT_PATH)
        or os.path.getsize(PROTOTXT_PATH) == 0
    ):
        response = requests.get(PROTOTXT_URL, timeout=60)
        response.raise_for_status()

        with open(PROTOTXT_PATH, "wb") as file:
            file.write(response.content)

    if (
        not os.path.isfile(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) == 0
    ):
        response = requests.get(MODEL_URL, timeout=180)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)

    if os.path.getsize(PROTOTXT_PATH) < 1000:
        raise RuntimeError("The model configuration file is invalid.")

    if os.path.getsize(MODEL_PATH) < 1000000:
        raise RuntimeError("The downloaded face model is incomplete.")

    detector = cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
    )

    return detector


def detect_faces(image_array, detector):
    height, width = image_array.shape[:2]
    result = image_array.copy()

    blob = cv2.dnn.blobFromImage(
        image_array,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    detector.setInput(blob)
    detections = detector.forward()

    candidates = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        box = (
            detections[0, 0, i, 3:7]
            * np.array([width, height, width, height])
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < MIN_FACE_SIZE or box_height < MIN_FACE_SIZE:
            continue

        ratio = box_width / box_height

        if ratio < 0.5 or ratio > 1.5:
            continue

        candidates.append(
            (x1, y1, x2, y2, confidence)
        )

    # Remove heavily overlapping duplicate detections.
    kept = []

    for candidate in sorted(
        candidates,
        key=lambda item: item[4],
        reverse=True
    ):
        x1, y1, x2, y2, confidence = candidate
        duplicate = False

        for existing in kept:
            ex1, ey1, ex2, ey2, _ = existing

            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            intersection = (
                max(0, ix2 - ix1)
                * max(0, iy2 - iy1)
            )

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (ex2 - ex1) * (ey2 - ey1)
            union = area1 + area2 - intersection

            overlap = intersection / union if union else 0

            if overlap > 0.45:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    for index, (x1, y1, x2, y2, confidence) in enumerate(
        kept, start=1
    ):
        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f"Face {index} ({confidence:.0%})"

        cv2.putText(
            result,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    return result, len(kept)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        st.subheader("Original Image")
        st.image(
            image_array,
            caption="Uploaded photograph",
            use_container_width=True
        )

        if st.button("🔍 Detect Faces", type="primary"):
            with st.spinner("Loading model and detecting faces..."):
                detector = load_face_detector()

                result, face_count = detect_faces(
                    image_array,
                    detector
                )

            if face_count:
                st.success(f"✅ Faces Detected: {face_count}")
            else:
                st.warning(
                    "No faces detected with sufficient confidence. "
                    "Try a clear photograph with faces facing the camera."
                )

            st.subheader("Detection Result")
            st.image(
                result,
                caption="Potential faces outlined in green",
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
        st.error("Face detection failed.")
        st.exception(error)

else:
    st.info("Upload a photograph to begin face detection.")

st.divider()

st.caption(
    "Face Detection AI | Python • Streamlit • OpenCV DNN"
)
