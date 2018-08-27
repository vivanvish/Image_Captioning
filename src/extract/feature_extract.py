import argparse
import os
import pickle
import queue
import sys
import threading
import zipfile

sys.path.append("..")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras

from model.encoder import Encoder


def get_img_embeddings(data, encoder, preprocessor, img_shape=(250, 250),
                       batch_size=32):
    """
        Creates image embeddings of the images in the data file (zip).
        Uses threading to achieve faster processing.
    """
    q = queue.Queue(maxsize=batch_size*10)
    read_done_event = threading.Event()
    stop_read_event = threading.Event()

    def img_batch_read(data):
        data_zip = zipfile.ZipFile(data)
        for img_name in data_zip.namelist():
            if stop_read_event.is_set():
                break
            if '.jpg' in img_name:
                img_bytes = data_zip.read(img_name)
                # Getting image from bytes
                img = cv2.imdecode(np.asarray(bytearray(img_bytes),
                                              dtype=np.uint8), 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = _preprocess_img(img, img_shape, preprocessor)
                # Adding image to queue
                while True:
                    try:
                        q.put((os.path.split(img_name)[-1], img), timeout=1)
                    except queue.Full:
                        if stop_read_event.is_set():
                            break
                        continue
                    break
        print('Data processing done.')
        read_done_event.set()
    daemon = threading.Thread(target=img_batch_read, args=(data,))
    daemon.daemon = True
    daemon.start()

    files = []
    img_embeddings = []
    batch = []

    def generate_embeddings(batch):
        batch = np.stack(batch)
        embeddings = encoder.predict(batch)
        img_embeddings.append(embeddings)

    try:
        while True:
            try:
                file, img = q.get(timeout=1)
            except queue.Empty:
                if read_done_event.is_set():
                    break
                continue
            files.append(file)
            batch.append(img)
            if len(batch) == batch_size:
                generate_embeddings(batch)
                batch = []
            q.task_done()
        if len(batch):
            generate_embeddings(batch)
    finally:
        stop_read_event.set()
        daemon.join()

    q.join()
    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, files


def _preprocess_img(img, img_shape, preprocessor):
    """
        Function to crop, resize and perform model pre-processing on the
        input data.
    """
    img = _crop_img(img)
    img = cv2.resize(img, img_shape)
    img = img.astype("float32")
    img = preprocessor(img)
    return img


def _crop_img(img):
    """
    Crops and returns a square image.
    """
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]


def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process params.')
    parser.add_argument(
        "--input-data", type=str, required=True,
        help="The full path of the zipfile with input data.")
    parser.add_argument(
        "--dest-folder", type=str, required=True,
        help="The full path of where to save the embeddings.")
    parser.add_argument(
        "--dest-prepend", type=str, required=True,
        help="The string to prepend on the file name.")
    parser.add_argument(
        "--crop-shape", type=int, default=299,
        help="The image crop shape.")
    parser.add_argument(
        "--batch-size", type=int, required=True,
        help="The batch size for processing images.")
    args = parser.parse_args()
    encoder, preprocessor = Encoder().get_cnn_encoder_preprocessor()
    print('----------Starting feature extraction from: {}'.format(
        args.input_data))

    img_embeds, files = get_img_embeddings(
        args.input_data, encoder, preprocessor, img_shape=(args.crop_shape,
                                                           args.crop_shape))
    print('----------Feature extraction done!. Saving data to: {}'.format(
        args.dest_folder))
    save_pickle(img_embeds, args.dest_folder +
                "/{}_img_embeds.pickle".format(args.dest_prepend))
    save_pickle(files, args.dest_folder +
                "/{}_img_files.pickle".format(args.dest_prepend))
    print("Done. Exiting Process...")
