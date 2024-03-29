import io
from enum import Enum
from typing import Type
import onnxruntime as ort
import numpy as np
from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
)
from PIL import Image
from PIL.Image import Image as PILImage
import pooch
import os
from pathlib import Path

class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2

def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

def post_process(mask):
    mask = morphologyEx(mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, (3, 3)))
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)
    return mask


def remove(data, post_process_mask=False) :

    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    session = new_session("u2net")

    masks = session.predict(img)
    mask = masks[0]

    if post_process_mask:
        mask = Image.fromarray(post_process(np.array(mask)))
    cutout = naive_cutout(img, mask)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    return cutout

# onnx session
class BaseSession:
    def __init__(self, model_name, inner_session):
        self.model_name = model_name
        self.inner_session = inner_session

    def normalize(self, img, mean, std,size):
        im = img.convert("RGB").resize(size, Image.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img):
        raise NotImplementedError

def new_session(model_name = "u2net"):
    session_class: Type[BaseSession]
    md5 = "60024c5c889badc19c04ad937298a77b"
    url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    session_class = SimpleSession

    u2net_home = os.getenv(
        "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    )

    fname = f"{model_name}.onnx"
    path = Path(u2net_home).expanduser()
    full_path = Path(u2net_home).expanduser() / fname

    pooch.retrieve(
        url,
        f"md5:{md5}",
        fname=fname,
        path=Path(u2net_home).expanduser(),
        progressbar=True,
    )

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(
        model_name,
        ort.InferenceSession(
            str(full_path),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        ),
    )

class SimpleSession(BaseSession):
    def predict(self, img):
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]
