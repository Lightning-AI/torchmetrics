# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import csv
import gzip
import os
import pathlib
import pickle
import shutil
import sys
import urllib
import warnings
from functools import partial

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import linalg
from tqdm import tqdm, tqdm_notebook

model_params = {
    'Inception': {
        'name': 'Inception',
        'imsize': 64,
        'output_layer': 'Pretrained_Net/pool_3:0',
        'input_layer': 'Pretrained_Net/ExpandDims:0',
        'output_shape': 2048,
        'cosine_distance_eps': 0.1
        },
    'NASNet': {
        'name': 'NASNet',
        'imsize': 331,
        'output_layer': 'Pretrained_Net/final_layer/Mean:0',
        'input_layer': 'Pretrained_Net/input:0',
        'output_shape': 4032,
        'cosine_distance_eps': 0.25 # tmp for now.
        }
}

def preprocessing(img, size, model):
    '''
    src: https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/preprocessing_factory.py
         https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    '''
    # resize.
    img = tf.expand_dims(img, 0)
    scaled_img = tf.image.resize_bilinear(
        img,
        [size, size],
        align_corners=False,
    )
    if model == 'NASNet':
        # fit range to [-1, 1]
        normalized_img = (scaled_img / 255)
        normalized_img -= 0.5
        normalized_img *= 2
        return normalized_img[0]
    elif model == 'Inception:
        return scaled_img[0]
    else:
        raise NotImplementedError

def create_model_graph(pth):
    """Creates a graph from saved GraphDef file."""
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')

def get_model_layer(sess, model_name):
    layername = model_params[model_name]['output_layer']
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return layer

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    return image

def get_activations(image_files, sess, model_name, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- image_files : Name of image files.
                     Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """

    n_images = len(image_files)
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    partial_preprocessing = partial(preprocessing, size=model_params[model_name]['imsize'], model=model_name)

    # create dataset
    paths_ds = tf.data.Dataset.from_tensor_slices(image_files)
    image_ds = paths_ds.map(load_image).map(partial_preprocessing)
    image_ds = image_ds.batch(batch_size).prefetch(5)

    img_tensor = image_ds.make_one_shot_iterator().get_next()
    target_layer = get_model_layer(sess, model_name)
    target_layer_flat = tf.reshape(target_layer, [-1, model_params[model_name]['output_shape']])

    pred_arr = np.zeros([n_images, model_params[model_name]['output_shape']], dtype=np.float64)
    head = 0
    end = 0

    while True:
        try:
            images = sess.run(img_tensor)
            feats = sess.run(target_layer_flat, {model_params[model_name]['input_layer']: images})
            true_bsize = feats.shape[0]
            end += true_bsize
            pred_arr[head: end] = feats
            head += true_bsize
        except tf.errors.OutOfRangeError:
            break

    if verbose:
        print(" done")

    return pred_arr


def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

def cosine_distance(features1, features2):
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    mean_min_d = np.mean(np.min(d, axis=1))
    return mean_min_d

def distance_thresholding(d, eps):
    if d < eps:
        return d
    else:
        return 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise Exception("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(image_files, sess, model_name, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- image_files : filenames of .png images
                     Numpy array of dimension (n_images, ?, ?, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(image_files, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act

def handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):

    if path.endswith('.npz'):
        f = np.load(path)
        m, s, features = f['m'][:], f['s'][:], f['features']
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        imsize = model_params[model_name]['imsize']

        if model_name == 'NASNet': # only resize when
            is_checksize = False

        filenames = [img_read_checks(f, is_checksize, imsize, is_check_png) for f in files]

        m, s, features = calculate_activation_statistics(filenames, sess, model_name)

    return m, s, features

def img_read_checks(filename, is_checksize=False, check_imsize = 64, is_check_png = False):

    with Image.open(str(filename)) as im:
        if is_checksize and im.size != (check_imsize,check_imsize):
            raise Exception('The images are not of size '+str(check_imsize))

        if is_check_png and im.format != 'PNG':
            raise Exception('Only PNG images should be submitted.')
    return str(filename)

def calculate_mifid(gen_image_dir, model_name, model_path, feature_path, verbose=False):
    ''' Calculates the MiFID'''
    tf.reset_default_graph()
    create_model_graph(str(model_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1, features1 = handle_path_memorization(gen_image_dir, sess, model_name,
                                                     is_checksize=True, is_check_png=True)

        if os.path.isdir(feature_path): # no pre-trained features, then get features from training images
            m2, s2, features2 = handle_path_memorization(feature_path, sess, model_name,
                                                         is_checksize=False, is_check_png=False)
        else:
            with np.load(feature_path) as f:
                m2, s2, features2 = f['m'], f['s'], f['features']

    if verbose:
        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))
        print('starting calculating FID')
    fid = calculate_frechet_distance(m1, s1, m2, s2)
    if verbose:
        print('done with FID, starting distance calculation')
    distance = cosine_distance(features1, features2)
    return fid, distance

def main():
    if len(sys.argv) != 2:
        raise Exception("Only accept one argument: directory path of generated images.")
    gen_image_dir = sys.argv[1] # expect generated pngs in this directory

    public_model_path = './data/public_inception_model.pb'
    private_model_path = './data/private_nasnet_model.pb'

    public_feat_path = './data/public_inception_feats.npz'
    private_feat_path = './data/private_nasnet_feats.npz'

    fid_epsilon = 1e-15

    fid_public, distance_public = calculate_mifid(gen_image_dir, 'Inception',
                                                  public_model_path, public_feat_path)

    distance_public_thresholded = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])
    public_score = fid_public / (distance_public_thresholded + fid_epsilon)

    print("FID_public: ", fid_public, "distance_public: ", distance_public, "public_score: ", public_score)

    fid_private, distance_private = calculate_mifid(gen_image_dir, 'NASNet',
                                                    private_model_path, private_feat_path)

    distance_private_thresholded = distance_thresholding(distance_private, model_params['NASNet']['cosine_distance_eps'])
    private_score = fid_private / (distance_private_thresholded + fid_epsilon)

    print("FID_private: ", fid_private, "distance_private: ", distance_private, "private_score: ", private_score)

if __name__ == '__main__':
    main()
