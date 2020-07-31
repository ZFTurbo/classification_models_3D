import functools
import keras_applications as ka

from .models import resnet as rn
from .models import resnext as rx
from .models import senet as sn
from .models import densenet as dn
from .models import vgg16 as vgg16
from .models import vgg19 as vgg19
from .models import inception_resnet_v2 as irv2
from .models import inception_v3 as iv3
from .models import mobilenet as mb1
from .models import mobilenet_v2 as mb2


class ModelsFactory:
    _models = {

        # ResNets
        'resnet18': [rn.ResNet18, rn.preprocess_input],
        'resnet34': [rn.ResNet34, rn.preprocess_input],
        'resnet50': [rn.ResNet50, rn.preprocess_input],
        'resnet101': [rn.ResNet101, rn.preprocess_input],
        'resnet152': [rn.ResNet152, rn.preprocess_input],

        # SE-Nets
        'seresnet18': [rn.SEResNet18, rn.preprocess_input],
        'seresnet34': [rn.SEResNet34, rn.preprocess_input],
        'seresnet50': [sn.SEResNet50, sn.preprocess_input],
        'seresnet101': [sn.SEResNet101, sn.preprocess_input],
        'seresnet152': [sn.SEResNet152, sn.preprocess_input],
        'seresnext50': [sn.SEResNeXt50, sn.preprocess_input],
        'seresnext101': [sn.SEResNeXt101, sn.preprocess_input],
        'senet154': [sn.SENet154, sn.preprocess_input],

        # ResNext
        'resnext50': [rx.ResNeXt50, rx.preprocess_input],
        'resnext101': [rx.ResNeXt101, rx.preprocess_input],

        # VGG
        'vgg16': [vgg16.VGG16, vgg16.preprocess_input],
        'vgg19': [vgg19.VGG19, vgg19.preprocess_input],

        # Densnet
        'densenet121': [dn.DenseNet121, dn.preprocess_input],
        'densenet169': [dn.DenseNet169, dn.preprocess_input],
        'densenet201': [dn.DenseNet201, dn.preprocess_input],

        # Inception
        'inceptionresnetv2': [irv2.InceptionResNetV2, irv2.preprocess_input],
        'inceptionv3': [iv3.InceptionV3, iv3.preprocess_input],

        # MobileNet
        'mobilenet': [mb1.MobileNet, mb1.preprocess_input],
        'mobilenetv2': [mb2.MobileNetV2, mb2.preprocess_input],
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    @staticmethod
    def get_kwargs():
        return {}

    def inject_submodules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            modules_kwargs = self.get_kwargs()
            new_kwargs = dict(list(kwargs.items()) + list(modules_kwargs.items()))
            return func(*args, **new_kwargs)

        return wrapper

    def get(self, name):
        if name not in self.models_names():
            raise ValueError('No such model `{}`, available models: {}'.format(
                name, list(self.models_names())))

        model_fn, preprocess_input = self.models[name]
        model_fn = self.inject_submodules(model_fn)
        preprocess_input = self.inject_submodules(preprocess_input)
        return model_fn, preprocess_input
