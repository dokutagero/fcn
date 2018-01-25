# -*- coding: utf-8 -*-

import chainer
import cv2
import fcn
import numpy as np
import os
import skimage.io
import time

from flask_uploads import (
    UploadSet,
    configure_uploads,
    IMAGES,
    patch_request_class
)

from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from io import BytesIO
from PIL import Image
from wtforms import SubmitField
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
app.config['SECRET_KEY'] = '24e3013ee601617ed2810ce46e7356bba2d2823f5231cf8d'
app.config['UPLOADED_PHOTOS_DEST'] = '/root/fcn/juanjo_fcn/examples/bridge/infer_web/uploads/'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
# set maximum file size, default is 16MB
patch_request_class(app)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, u'Allow image only.'),
            FileRequired(u'File was empty.')])
    submit = SubmitField(u'Upload')


def infer(im):
    model_file = '/root/fcn/juanjo_fcn/examples/bridge/logs/with_aug/fcn8s_VCS-2ebd26d_TIME-20180119-232614/models/FCN8s_iter00095608.npz'
    im_np = np.array(im)
    model_class = getattr(fcn.models, 'FCN8s')
    model = model_class(n_class=3)
    chainer.serializers.load_npz(model_file, model)
    chainer.cuda.get_device(0).use()
    model.to_gpu()
    label_names = fcn.datasets.BridgeSeg.class_names
    input, = fcn.datasets.transform_lsvrc2012_vgg16((im_np,))
    input = input[np.newaxis, :, :, :]
    input = chainer.cuda.to_gpu(input)
    with chainer.no_backprop_mode():
        input = chainer.Variable(input)
        with chainer.using_config('train', False):
            model(input)
            lbl_pred = chainer.functions.argmax(model.score, axis=1)[0]
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
    viz = fcn.utils.label2rgb(lbl_pred, img=im_np, label_names=label_names,
                  n_labels=3)

    # Return image back
    image = Image.fromarray(viz)
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)

    return byte_io


@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        try:
            os.remove(
                app.config['UPLOADED_PHOTOS_DEST'] + 'out.png'
            )
        except OSError:
            pass
        im = Image.open(BytesIO(form.photo.data.read()))
        filename = photos.save(
            FileStorage(
                stream=infer(im),
                filename='out.png'
            )
        )
        file_url = photos.url(filename)
        file_url += '?{}'.format(int(time.time()))
    else:
        file_url = None

    return render_template(
        'index.html',
        form=form,
        file_url=file_url
    )
