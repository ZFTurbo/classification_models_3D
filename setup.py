try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='classification_models_3D',
    version='1.0.8',
    author='Roman Sol (ZFTurbo)',
    packages=['classification_models_3D', 'classification_models_3D/models'],
    url='https://github.com/ZFTurbo/classification_models_3D',
    description='Set of models for classification of 3D volumes.',
    long_description='3D variants of popular CNN models for classification like ResNets, DenseNets, VGG, etc. '
                     'It also contains weights obtained by converting ImageNet weights from the same 2D models. '
                     'Models work with keras and tensorflow.keras.'
                     'More details: https://github.com/ZFTurbo/classification_models_3D',
)
