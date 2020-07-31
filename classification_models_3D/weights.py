from . import get_submodules_from_kwargs

__all__ = ['load_model_weights']


def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):
    _, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = keras_utils.get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


WEIGHTS_COLLECTION = [

    # resnet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet18_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet18_inp_channel_3_tch_0_top_False.h5',
        'md5': 'e616829b530e021857ccf5ff02cf83a0',
    },
    # resnet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet18_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnet18_inp_channel_3_tch_0_top_True.h5',
        'md5': '1ebbd4226330d7f21ddb5a0e93ab78d7',
    },
    # resnet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet34_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet34_inp_channel_3_tch_0_top_False.h5',
        'md5': '43dfb56dc89245b36a54e8322c4ca657',
    },
    # resnet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet34_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnet34_inp_channel_3_tch_0_top_True.h5',
        'md5': 'a3944515051370ab3e22b12f054083dd',
    },
    # resnet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet50_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet50_inp_channel_3_tch_0_top_False.h5',
        'md5': '7ffd82b2dbbc5167c5c7b708adb1f84d',
    },
    # resnet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet50_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnet50_inp_channel_3_tch_0_top_True.h5',
        'md5': 'a541bae9844dca5894235b9a63ddd0ff',
    },
    # resnet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet101_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet101_inp_channel_3_tch_0_top_False.h5',
        'md5': 'fbcfac63054199edfb99bda504122721',
    },
    # resnet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet101_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnet101_inp_channel_3_tch_0_top_True.h5',
        'md5': 'eae60e94ba8a5028e33cc4c4991f71a3',
    },
    # resnet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet152_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet152_inp_channel_3_tch_0_top_False.h5',
        'md5': 'cdc49b6f9fb60f1ee50d27361e83c91b',
    },
    # resnet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnet152_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnet152_inp_channel_3_tch_0_top_True.h5',
        'md5': 'ea1aafaadf81195e2a7f8943fce5eb9e',
    },
    # seresnet18
    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet18_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet18_inp_channel_3_tch_0_top_False.h5',
        'md5': 'd5604a96dfcea542432a7291a09bff62',
    },
    # seresnet18
    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet18_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnet18_inp_channel_3_tch_0_top_True.h5',
        'md5': 'c63735b19454b970db48309e8c37c1c3',
    },
    # seresnet34
    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet34_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet34_inp_channel_3_tch_0_top_False.h5',
        'md5': 'a1e9667a295923146ac44b382d783af2',
    },
    # seresnet34
    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet34_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnet34_inp_channel_3_tch_0_top_True.h5',
        'md5': 'ab747861f2cc5ff2467f600a48e134b7',
    },
    # seresnet50
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet50_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet50_inp_channel_3_tch_0_top_False.h5',
        'md5': '764aa40be1bd3973e4a0e139f05b46ec',
    },
    # seresnet50
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet50_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnet50_inp_channel_3_tch_0_top_True.h5',
        'md5': 'a830957ba7410853e96fe96978e7ce21',
    },
    # seresnet101
    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet101_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet101_inp_channel_3_tch_0_top_False.h5',
        'md5': '80b285852a7c32272528c48d3fe45e7e',
    },
    # seresnet101
    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet101_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnet101_inp_channel_3_tch_0_top_True.h5',
        'md5': '7897388bc7db10c914f7c021dd995305',
    },
    # seresnet152
    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet152_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet152_inp_channel_3_tch_0_top_False.h5',
        'md5': '4a2c0e17d87c0a4538208e742d9bfcf1',
    },
    # seresnet152
    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnet152_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnet152_inp_channel_3_tch_0_top_True.h5',
        'md5': '64715473de989e45c7e07e203d9fbf93',
    },
    # seresnext50
    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnext50_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnext50_inp_channel_3_tch_0_top_False.h5',
        'md5': '0bf5fd7dd6583ab29aba2584776048f1',
    },
    # seresnext50
    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnext50_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnext50_inp_channel_3_tch_0_top_True.h5',
        'md5': '5b4e167f0094e143b42507984d281059',
    },
    # seresnext101
    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnext101_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnext101_inp_channel_3_tch_0_top_False.h5',
        'md5': '1aa145227e62162a7c2da729023df2fb',
    },
    # seresnext101
    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/seresnext101_inp_channel_3_tch_0_top_True.h5',
        'name': 'seresnext101_inp_channel_3_tch_0_top_True.h5',
        'md5': '47cf612daa62614b2411d62d1fc5b04a',
    },
    # senet154
    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/senet154_inp_channel_3_tch_0_top_False.h5',
        'name': 'senet154_inp_channel_3_tch_0_top_False.h5',
        'md5': 'c74f8adfb4ff8f13923ec06d865831cd',
    },
    # senet154
    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/senet154_inp_channel_3_tch_0_top_True.h5',
        'name': 'senet154_inp_channel_3_tch_0_top_True.h5',
        'md5': '8ceb38a898bf88604787d768a20c8973',
    },
    # resnext50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnext50_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnext50_inp_channel_3_tch_0_top_False.h5',
        'md5': '2eb378806a7ac8792084155a8647b682',
    },
    # resnext50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnext50_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnext50_inp_channel_3_tch_0_top_True.h5',
        'md5': 'e683651b078481aec27003738770c4b3',
    },
    # resnext101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnext101_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnext101_inp_channel_3_tch_0_top_False.h5',
        'md5': 'f03aa41185a001558d62ecd1bc817935',
    },
    # resnext101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/resnext101_inp_channel_3_tch_0_top_True.h5',
        'name': 'resnext101_inp_channel_3_tch_0_top_True.h5',
        'md5': '3d1f46232206d9abeba3dfe74e3a5295',
    },
    # vgg16
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/vgg16_inp_channel_3_tch_0_top_False.h5',
        'name': 'vgg16_inp_channel_3_tch_0_top_False.h5',
        'md5': '81c6d51dd04e2735e0b9a4a81689ca75',
    },
    # vgg16
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/vgg16_inp_channel_3_tch_0_top_True.h5',
        'name': 'vgg16_inp_channel_3_tch_0_top_True.h5',
        'md5': '905fcec031dec3210042da9ebf49807a',
    },
    # vgg19
    {
        'model': 'vgg19',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/vgg19_inp_channel_3_tch_0_top_False.h5',
        'name': 'vgg19_inp_channel_3_tch_0_top_False.h5',
        'md5': '0733d91ae6559c497402245a2155b042',
    },
    # vgg19
    {
        'model': 'vgg19',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/vgg19_inp_channel_3_tch_0_top_True.h5',
        'name': 'vgg19_inp_channel_3_tch_0_top_True.h5',
        'md5': '533852c12c4569c0c58ce0b3a4afb581',
    },
    # densenet121
    {
        'model': 'densenet121',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet121_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet121_inp_channel_3_tch_0_top_False.h5',
        'md5': '743ea52b43c19000d9c4dcd328fd3f9d',
    },
    # densenet121
    {
        'model': 'densenet121',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet121_inp_channel_3_tch_0_top_True.h5',
        'name': 'densenet121_inp_channel_3_tch_0_top_True.h5',
        'md5': 'c9dec0d11eda5fb3ca85369849dbdc6c',
    },
    # densenet169
    {
        'model': 'densenet169',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet169_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet169_inp_channel_3_tch_0_top_False.h5',
        'md5': 'c295e36f1256299ccd36eff3ff8f8d0c',
    },
    # densenet169
    {
        'model': 'densenet169',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet169_inp_channel_3_tch_0_top_True.h5',
        'name': 'densenet169_inp_channel_3_tch_0_top_True.h5',
        'md5': '4a196e2a132d0e99d92f355a26d1d644',
    },
    # densenet201
    {
        'model': 'densenet201',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet201_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet201_inp_channel_3_tch_0_top_False.h5',
        'md5': 'bb2cedb308121480841711da50598f05',
    },
    # densenet201
    {
        'model': 'densenet201',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet201_inp_channel_3_tch_0_top_True.h5',
        'name': 'densenet201_inp_channel_3_tch_0_top_True.h5',
        'md5': '8f092e1e10a9fd79b79ef12bcb680a56',
    },
    # mobilenet
    {
        'model': 'mobilenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/mobilenet_inp_channel_3_tch_0_top_False.h5',
        'name': 'mobilenet_inp_channel_3_tch_0_top_False.h5',
        'md5': 'ff538af9bc4c9b8963d7105893c4a884',
    },
    # mobilenet
    {
        'model': 'mobilenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/mobilenet_inp_channel_3_tch_0_top_True.h5',
        'name': 'mobilenet_inp_channel_3_tch_0_top_True.h5',
        'md5': 'a4ef23a701c6747dbd78dc5eff89a5cc',
    },
    # mobilenetv2
    {
        'model': 'mobilenetv2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/mobilenetv2_inp_channel_3_tch_0_top_False.h5',
        'name': 'mobilenetv2_inp_channel_3_tch_0_top_False.h5',
        'md5': 'de2f84504762e9c3150d2aaf448fa378',
    },
    # mobilenetv2
    {
        'model': 'mobilenetv2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/mobilenetv2_inp_channel_3_tch_0_top_True.h5',
        'name': 'mobilenetv2_inp_channel_3_tch_0_top_True.h5',
        'md5': 'edf2cea91f5e343cd66c489c9811bdf0',
    },
]
