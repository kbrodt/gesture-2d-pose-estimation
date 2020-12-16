import urllib
from pathlib import Path

import cv2
import PIL.Image as Image
import requests
import streamlit as st

from utils import get_final_preds, get_input, put_kps

st.set_page_config(layout="wide")
ROOT = Path(".") / "assets"
DESIRED_SIZE = 512
DEFAULT_IMG_URL_PATH = (
    "https://drawpaintacademy.com/wp-content/uploads/2018/05/Michelangelo.jpg"
)
URL_MODEL = ''


@st.cache
def process_image(img_raw):
    pose_input, img, center, scale = get_input(img_raw)
    path_to_model = ROOT / "model_best.onnx" 
    if not path_to_model.exists():
        urllib.request.urlretrieve(URL_MODEL, path_to_model)

    model = cv2.dnn.readNetFromONNX(str(path_to_model))
    model.setInput(pose_input[None])
    predicted_heatmap = model.forward()
    predicted_keypoints, confidence = get_final_preds(
        predicted_heatmap, center[None], scale[None], post_process=True
    )

    predicted_keypoints, confidence, predicted_heatmap = (
        predicted_keypoints[0],
        confidence[0],
        predicted_heatmap[0],
    )

    img = Image.fromarray(img[..., ::-1])
    original_img_size = img.size
    ratio = float(DESIRED_SIZE) / max(original_img_size)
    resized_img_size = tuple([int(x * ratio) for x in original_img_size])
    img = img.resize(resized_img_size, Image.ANTIALIAS)
    original_img_size = max(original_img_size)

    predicted_keypoints *= (DESIRED_SIZE - 1) / (original_img_size - 1)

    predicted_heatmap = predicted_heatmap.sum(0)
    predicted_heatmap_min = predicted_heatmap.min()
    predicted_heatmap_max = predicted_heatmap.max()
    predicted_heatmap = (predicted_heatmap - predicted_heatmap_min) / (
        predicted_heatmap_max - predicted_heatmap_min
    )
    predicted_heatmap = Image.fromarray((predicted_heatmap * 255).astype("uint8"))
    predicted_heatmap = predicted_heatmap.resize(resized_img_size, Image.ANTIALIAS)

    return img, predicted_keypoints, confidence, predicted_heatmap


def main():
    st.title("2D pose estimation for sketches with human-like characters")

    inp = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if inp is None:
        st.warning("No file selected.")
        with open(ROOT / "test.png", "br") as inp2:
            img_raw = inp2.read()
    else:
        img_raw = inp.read()

    if st.checkbox("or use url"):
        url = st.text_input("The URL link", value=DEFAULT_IMG_URL_PATH)
        if url != "":
            img_raw = requests.get(url).content

    img, predicted_keypoints, confidence, predicted_heatmap = process_image(img_raw)

    thresh = st.slider("Confidence", 0.0, 1.0, value=0.0, step=0.05)
    img_with_keyponts, kps_json = put_kps(
        img, predicted_keypoints, confidence, thresh=thresh
    )

    st.image(
        [
            img,
            img_with_keyponts,
            predicted_heatmap,
        ],
        caption=["original", "annotated", "heatmap"],
    )

    st.write(
        "Model supports only one character per image. "
        "The character must be full length and be located in the middle."
    )

    if st.checkbox("Show keypoints in json format {'JOINT_NAME': [x, y, confidence]}"):
        st.write(kps_json)


if __name__ == "__main__":
    main()
