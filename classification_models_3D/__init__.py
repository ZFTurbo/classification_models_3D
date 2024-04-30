import keras


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', keras.backend)
    layers = kwargs.get('layers', keras.layers)
    models = kwargs.get('models', keras.models)
    utils = kwargs.get('utils', keras.utils)

    return backend, layers, models, utils
