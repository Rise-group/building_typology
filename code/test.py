import configparser
import os
from glob import glob

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import metrics
from keras.applications import VGG16, VGG19, InceptionV3, ResNet50, Xception
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers.core import Dense, Dropout
from keras.layers.merge import concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, fbeta_score
from tqdm import tqdm

from train import (get_cnn_model, get_combined_generator,
                   get_image_generator)



def get_report(generator, num_images, num_classes, batch_size, model, split, net_name, net_id, figures_dir):
    # Get predictions from model
    print(f'Predictions from {split}')

    y_true = np.array([])
    y_pred = np.array([])

    for step, (img_batch, batch_true) in tqdm(enumerate(generator)):
        if step >= num_images//batch_size:
            break
        batch_pred = model.predict_on_batch(img_batch)
        if step == 0:
            y_pred = batch_pred
            y_true = batch_true
        else:
            y_pred = np.vstack((y_pred, batch_pred))
            y_true = np.vstack((y_true, batch_true))

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    output_csv = pd.DataFrame({
        'true': y_true,
        'pred': y_pred})

    output_csv.to_csv(
        f'{figures_dir}/{net_id}{net_name}_{split}.csv', index=False)

    f2_score_beta = fbeta_score(y_true, y_pred, beta=2, average='micro')

    if num_classes == 8:
        classes = ['CR_LDUAL', 'CR_LINF_DNO', 'CR_LINF_DUC',
                   'CR_LWAL', 'MCF', 'MCF_DNO', 'MR', 'MUR']

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.title(f'{net_id}{net_name}_{split} -> f2={f2_score_beta:.3}')
    plt.savefig(f'{figures_dir}/{net_id}{net_name}_{split}.png')


def test_on_images(network, images_dir, models_dir, *args):
    """"""
    print('Testing on images...')
    # Extract parameters from args
    img_width, img_height, batch_size, models_dir, figures_dir = args

    # Get image generators
    num_images_val, num_classes_val, val_gen = get_image_generator(
        network, images_dir, 'val', img_width, img_height, batch_size)

    num_images_test, num_classes_test, test_gen = get_image_generator(
        network, images_dir, 'test', img_width, img_height, batch_size)

    # Make sure val/test have the same number of classes
    assert num_classes_val == num_classes_test

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes_val, activation='softmax')(x)
    model = Model(base_model.input, outputs=predictions)

    path = f'{models_dir}/{network}'
    models_list = glob(f'{path}/A*', )

    net_id = f'{os.path.basename(models_dir)}A_'
    print('Loading model...')
    for h5_model in models_list:
        if gpu_number > 1:
            multi_gpu_model = load_model(
                h5_model,
                custom_objects={
                    'categorical_f1_score': km.categorical_f1_score()
                })
            model = multi_gpu_model.layers[-2]
        else:
            model.load_weights(h5_model)
        print('Validation')
        get_report(val_gen, num_images_val, num_classes_val, batch_size,
                   model, 'val', network, net_id, figures_dir)
        print('Test')
        get_report(test_gen, num_images_test, num_classes_test, batch_size,
                   model, 'test', network, net_id, figures_dir)


def test_combined(network, images_dir, csv_dir, csv_index, models_dir, *args):
    # Extract parameters from args
    img_width, img_height, batch_size, models_dir, figures_dir = args

    # Get image generators

    num_images_val, num_classes_val, features, val_gen = get_combined_generator(
        network, images_dir, csv_dir, csv_index, 'val', img_width, img_height, batch_size)

    num_images_test, num_classes_test, features, test_gen = get_combined_generator(
        network, images_dir, csv_dir, csv_index, 'test', img_width, img_height, batch_size)

    # Make sure val/test have the same number of classes
    assert num_classes_val == num_classes_test

    # Get network model for an image input shape
    input_shape = (img_width, img_height, 3)
    main_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, main_input)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)

    # Load Simple MLP
    aux_input = Input(shape=(features, ))
    aux = Dense(512, activation='relu')(aux_input)
    aux = Dropout(0.3)(aux)
    aux = Dense(512, activation='relu')(aux)
    aux = Dropout(0.3)(aux)
    aux = Dense(512, activation='relu')(aux)

    # Merge input models
    merge = concatenate([x, aux])
    predictions = Dense(num_classes_val, activation='softmax')(merge)
    
    model = Model(inputs=[main_input, aux_input], outputs=predictions)

    path = f'{models_dir}/{network}'
    models_list = glob(f'{path}/B*', )
    net_id = f'{os.path.basename(models_dir)}B_'

    for h5_model in models_list:
        if gpu_number > 1:
            multi_gpu_model = load_model(
                h5_model,
                custom_objects={
                    'categorical_f1_score': km.categorical_f1_score()
                },
            )
            model = multi_gpu_model.layers[-2]
        else:
            model.load_weights(h5_model)

        get_report(val_gen, num_images_val, num_classes_val,
                   batch_size, model, 'val', network, net_id, figures_dir)
        get_report(test_gen, num_images_val, num_classes_val,
                   batch_size, model, 'test', network, net_id, figures_dir)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read image parameters
    images_dir = config.get('IMAGES', 'images_dir')
    img_width = config.getint('IMAGES', 'width')
    img_height = config.getint('IMAGES', 'height')
    # Read csv parameters
    csv_dir = config.get('CSV', 'csv_dir')
    csv_index = config.get('CSV', 'csv_index').split(',')
    # Read training parameters
    lr_rate = config.getfloat('TRAINING', 'lr_rate')
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    networks_list = config.get('TRAINING', 'cnn_network_list').split(',')
    gpu_number = config.getint('TRAINING', 'gpu_number')

    # batch_size = batch_size * gpu_number

    # Read data paths
    models_dir = config.get('OUTPUT', 'models_dir')
    logs_dir = config.get('OUTPUT', 'logs_dir')
    figures_dir = config.get('OUTPUT', 'figures_dir')
    os.makedirs(f'{figures_dir}', exist_ok=True)

    for network in networks_list:
        K.clear_session()
        args = [img_width, img_height, batch_size, models_dir, figures_dir]

        test_on_images(network, images_dir, models_dir, *args)
        test_combined(network, images_dir, csv_dir,
                      csv_index, models_dir, *args)
