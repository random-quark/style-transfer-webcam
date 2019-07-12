from __future__ import print_function
from __future__ import division

import os
import pdb
import vgg
import transform
from datetime import datetime
import cv2
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.insert(0, 'src')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

models_all = [{"ckpt": "models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style": "styles/cubist-landscape-justineivu-geanina.jpg"},
              {"ckpt": "models/ckpt_hokusai_b20_e4_cw15/fns.ckpt",
                  "style": "styles/hokusai.jpg"},
              {"ckpt": "models/wave/wave.ckpt", "style": "styles/hokusai.jpg"},
              {"ckpt": "models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt",
               "style": "styles/kandinsky2.jpg"},
              {"ckpt": "models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt",
               "style": "styles/liechtenstein.jpg"},
              {"ckpt": "models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt",
               "style": "styles/maps3.jpg"},
              {"ckpt": "models/ckpt_wu_b20_e4_cw15/fns.ckpt",
                  "style": "styles/wu4.jpg"},
              {"ckpt": "models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt",
               "style": "styles/elsalahi2.jpg"},
              {"ckpt": "models/scream/scream.ckpt",
                  "style": "styles/the_scream.jpg"},
              {"ckpt": "models/udnie/udnie.ckpt", "style": "styles/udnie.jpg"},
              {"ckpt": "models/ckpt_clouds_b5_e2_cw05_tv1_04/fns.ckpt", "style": "styles/clouds.jpg"}]


models = [
    {"ckpt": "models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt",
     "style": "styles/kandinsky2.jpg"}, # not bad, quite plain
    {"ckpt": "models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style": "styles/wu4.jpg"}, # cool!
    {"ckpt": "models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt",
     "style": "styles/elsalahi2.jpg"},  # nice
    {"ckpt": "models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt",
        "style": "styles/liechtenstein.jpg"}, # quite good

    {"ckpt": "models/ckpt_cubist_b20_e4_cw05/fns.ckpt",
     "style": "styles/cubist-landscape-justineivu-geanina.jpg"},  # not bad

    # NO
    {"ckpt": "models/scream/scream.ckpt", "style": "styles/the_scream.jpg"},
    {"ckpt": "models/udnie/udnie.ckpt", "style": "styles/udnie.jpg"},
    {"ckpt": "models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt",
        "style": "styles/maps3.jpg"},
    {"ckpt": "models/ckpt_hokusai_b20_e4_cw15/fns.ckpt",
        "style": "styles/hokusai.jpg"}]  # no, too distinctive

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int,
                    help='camera device id (default 0)', required=False, default=0)
parser.add_argument('--width', type=int,
                    help='width to resize camera feed to (default 320)', required=False, default=640)
parser.add_argument('--disp_width', type=int,
                    help='width to display output (default 640)', required=False, default=1200)
parser.add_argument('--disp_source', type=int,
                    help='whether to display content and style images next to output, default 1', required=False, default=1)
parser.add_argument('--horizontal', type=int,
                    help='whether to concatenate horizontally (1) or vertically(0)', required=False, default=1)
parser.add_argument('--num_sec', type=int,
                    help='number of seconds to hold current model before going to next (-1 to disable)', required=False, default=-1)


def load_checkpoint(checkpoint, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint)
        style = cv2.imread(checkpoint)
        return True
    except:
        print("checkpoint %s not loaded correctly" % checkpoint)
        return False


def get_camera_shape(cam):
    """ use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
    cv_version_major, _, _ = cv2.__version__.split('.')
    if cv_version_major == '3' or cv_version_major == '4':
        return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


def create_faces_mask(original_image, buffer):
    gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    shape = (original_image.shape[0], original_image.shape[1], 1)
    mask = np.full(shape, 0, dtype=np.uint8)
    faces = face_cascade.detectMultiScale(gray_original_image, 1.3, 5)
    face_areas = []
    for (x, y, w, h) in faces:
        face_areas.append(([x, y, w, h], original_image[y:y+h, x:x+w]))
        center = (x + w // 2, y + h // 2)
        radius = max(h, w) // 2 + buffer
        cv2.circle(mask, center, radius, (255), -1)
        mask = blur(mask, 50)
    return mask


def blur(image, blur_amount):
    kernel = np.ones((blur_amount, blur_amount), np.float32) / blur_amount ** 2
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


def create_composite(original_image, transformed_image, mask):
    mask_inverted = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(
        transformed_image, transformed_image, mask=mask_inverted)
    foreground = cv2.bitwise_and(original_image, original_image, mask=mask)
    composite = alphaBlend(background, foreground, mask)
    return composite


def main(device_id, width, disp_width, disp_source, horizontal, num_sec):
    t1 = datetime.now()
    idx_model = 0
    device_t = '/gpu:0'
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        cam = cv2.VideoCapture(device_id)
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
        cam_width, cam_height = get_camera_shape(cam)
        width = width if width % 4 == 0 else width + \
            4 - (width % 4)  # must be divisible by 4
        height = int(width * float(cam_height/cam_width))
        height = height if height % 4 == 0 else height + \
            4 - (height % 4)  # must be divisible by 4
        img_shape = (height, width, 3)
        batch_shape = (1,) + img_shape
        print("batch shape", batch_shape)
        print("disp source is ", disp_source)
        img_placeholder = tf.placeholder(
            tf.float32, shape=batch_shape, name='img_placeholder')
        preds = transform.net(img_placeholder)

        # load checkpoint
        load_checkpoint(models[idx_model]["ckpt"], sess)
        style = cv2.imread(models[idx_model]["style"])

        # enter cam loop
        while True:
            ret, frame = cam.read()
            frame = cv2.resize(frame, (width, height))
            frame = cv2.flip(frame, 1)

            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = frame

            output = sess.run(preds, feed_dict={img_placeholder: X})

            output = output[:, :, :, [2, 1, 0]].reshape(img_shape)
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.resize(output, (width, height))

            mask = create_faces_mask(frame, 15)
            composite = create_composite(frame, output, mask)

            # oh, ow, _ = output.shape
            # output = cv2.resize(
            #     output, (disp_width, int(oh * disp_width / ow)))
            # cv2.imshow('frame', output)

            oh, ow, _ = composite.shape
            composite = cv2.resize(
                composite, (disp_width, int(oh * disp_width / ow)))
            cv2.imshow('frame', composite)

            key_ = cv2.waitKey(1)

        # done
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts.device_id, opts.width, opts.disp_width,
         opts.disp_source == 1, opts.horizontal == 1, opts.num_sec),
