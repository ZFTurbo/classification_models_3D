# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import sys
    import os

    # For this test, make sure that only the tested framework is available
    sys.modules['tensorflow'] = None
    sys.modules['torch'] = None

    gpu_use = 0
    print(f"GPU use: {gpu_use}")
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_use}"


from classification_models_3D.kkeras import Classifiers
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import measure
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling3D
from keras.models import Model
from keras.src.utils import summary_utils


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model' or layer_type == 'Functional':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = summary_utils.count_params(model.trainable_weights)
    non_trainable_count = summary_utils.count_params(model.non_trainable_weights)

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0


def gen_random_volume(debug=False):
    img = np.zeros((96, 96, 96, 3), dtype=np.uint8)
    answ = np.array([0, 0])
    num_sheres = random.randint(2, 3)
    min_radius = 3
    max_radius = 20
    num_cubes = random.randint(2, 3)
    min_cube_side = 3
    max_cube_side = 15

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[..., 0] = dark_color0
    img[..., 1] = dark_color1
    img[..., 2] = dark_color2

    if random.uniform(0, 1) > 0.5:
        # Spheres
        for i in range(num_sheres):
            light_color0 = random.randint(dark_color0+1, 255)
            light_color1 = random.randint(dark_color1+1, 255)
            light_color2 = random.randint(dark_color2+1, 255)
            center_0 = random.randint(0, img.shape[0] - 1)
            center_1 = random.randint(0, img.shape[1] - 1)
            center_2 = random.randint(0, img.shape[2] - 1)
            r1 = random.randint(min_radius, max_radius)
            # print(r1, (center_0, center_1, center_2), (light_color0, light_color1, light_color2))
            s = sphere(img.shape[:-1], r1, (center_0, center_1, center_2))
            tmp = img.copy()
            tmp[s] = (light_color0, light_color1, light_color2)
            img[s] = tmp[s]
            # print(img.min(), img.max(), img.mean(), img.dtype)
        answ[0] = 1

    if random.uniform(0, 1) > 0.5:
        # Cubes
        for i in range(num_cubes):
            light_color0 = random.randint(dark_color0 + 1, 255)
            light_color1 = random.randint(dark_color1 + 1, 255)
            light_color2 = random.randint(dark_color2 + 1, 255)
            range0_start = random.randint(0, img.shape[0] - max_cube_side)
            range0_end = range0_start + random.randint(min_cube_side, max_cube_side)
            range1_start = random.randint(0, img.shape[1] - max_cube_side)
            range1_end = range1_start + random.randint(min_cube_side, max_cube_side)
            range2_start = random.randint(0, img.shape[2] - max_cube_side)
            range2_end = range2_start + random.randint(min_cube_side, max_cube_side)
            img[range0_start:range0_end, range1_start:range1_end, range2_start:range2_end] = (light_color0, light_color1, light_color2)
        answ[1] = 1

    # Debug
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        verts, faces, normals, values = measure.marching_cubes(img[..., 1], 127)
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            cmap='Spectral',
            antialiased=False,
            linewidth=0.0
        )
        plt.show()

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if random.random() < density:
                    img[i, j, k] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )

    return img, answ


def batch_generator(batch_size, preprocess_input):
    while True:
        image_list = []
        answ_list = []
        for i in range(batch_size):
            img, answ = gen_random_volume()
            image_list.append(img)
            answ_list.append(answ)

        image_list = np.array(image_list, dtype=np.float32)
        image_list = preprocess_input(image_list)
        answ_list = np.array(answ_list, dtype=np.float32)
        # print(image_list.shape, answ_list.shape)
        yield image_list, answ_list


def train_model_example():
    use_weights = 'imagenet'
    shape_size = (96, 96, 96, 3)
    backbone = 'resnet18'
    num_classes = 2
    batch_size_train = 12
    batch_size_valid = 12
    learning_rate = 0.0001
    patience = 10
    epochs = 50
    steps_per_epoch = 100
    validation_steps = 20
    dropout_val = 0.1

    modelPoint, preprocess_input = Classifiers.get(backbone)
    model = modelPoint(
        input_shape=shape_size,
        include_top=False,
        weights=use_weights,
    )
    x = model.layers[-1].output
    x = GlobalAveragePooling3D()(x)
    x = Dropout(dropout_val)(x)
    x = Dense(num_classes, name='prediction')(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=model.inputs, outputs=x)

    print(model.summary())
    print(get_model_memory_usage(batch_size_train, model))
    optim = Adam(learning_rate=learning_rate)

    loss_to_use = 'binary_crossentropy'
    model.compile(optimizer=optim, loss=loss_to_use, metrics=['acc',])

    cache_model_path = '{}_temp.keras'.format(backbone)
    best_model_path = '{}'.format(backbone) + '-{val_loss:.4f}-{epoch:02d}.keras'
    callbacks = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
        CSVLogger('history_{}_lr_{}.csv'.format(backbone, learning_rate), append=True),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min'),
    ]

    gen_train = batch_generator(
        batch_size_train,
        preprocess_input
    )
    gen_valid = batch_generator(
        batch_size_valid,
        preprocess_input,
    )
    history = model.fit(
        gen_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=gen_valid,
        validation_steps=validation_steps,
        verbose=1,
        initial_epoch=0,
        callbacks=callbacks
    )

    best_loss = max(history.history['val_loss'])
    print('Training finished. Loss: {}'.format(best_loss))


if __name__ == '__main__':
    # gen_random_volume(debug=True)
    train_model_example()
