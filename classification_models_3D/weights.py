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
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet18_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet18_inp_channel_3_tch_0_top_False.h5',
        'md5': '1d04dd6c1f00b7bf4ba883c61bedeac8',
    },
    # resnet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet34_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet34_inp_channel_3_tch_0_top_False.h5',
        'md5': 'b7f3bcdc67c8614ba018c9fc5f75fc64',
    },
    # resnet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet50_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet50_inp_channel_3_tch_0_top_False.h5',
        'md5': '2ba65fa189439a493ea8d1b22439ea2a',
    },
    # resnet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet101_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet101_inp_channel_3_tch_0_top_False.h5',
        'md5': 'fead4c03863fce1037fc35938b00ee56',
    },
    # resnet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet152_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet152_inp_channel_3_tch_0_top_False.h5',
        'md5': '4746b59c632b2e2d02d4abbceed605a7',
    },
    # seresnet18
    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnet18_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet18_inp_channel_3_tch_0_top_False.h5',
        'md5': 'e45098ba1f6294bd5a2f59678c39564b',
    },
    # seresnet34
    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnet34_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet34_inp_channel_3_tch_0_top_False.h5',
        'md5': 'c449f05f38819ecd429e277ccc6d8ea6',
    },
    # seresnet50
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnet50_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet50_inp_channel_3_tch_0_top_False.h5',
        'md5': 'ca85ba2de828a28a6dc26549d0711f9c',
    },
    # seresnet101
    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnet101_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet101_inp_channel_3_tch_0_top_False.h5',
        'md5': 'fffed2d93f2ca5aca2d583ebc889cdc0',
    },
    # seresnet152
    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnet152_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnet152_inp_channel_3_tch_0_top_False.h5',
        'md5': '5c5e7e1b3e79c2dbbfceb0029c492317',
    },
    # seresnext50
    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnext50_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnext50_inp_channel_3_tch_0_top_False.h5',
        'md5': 'd88b3039de2a974afaac0539944b06b2',
    },
    # seresnext101
    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/seresnext101_inp_channel_3_tch_0_top_False.h5',
        'name': 'seresnext101_inp_channel_3_tch_0_top_False.h5',
        'md5': '011bb535914e8d87bb6a9ed8e5e02124',
    },
    # senet154
    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/senet154_inp_channel_3_tch_0_top_False.h5',
        'name': 'senet154_inp_channel_3_tch_0_top_False.h5',
        'md5': '23db8cfa9d86359fa3b24a6a0abae423',
    },
    # resnext50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnext50_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnext50_inp_channel_3_tch_0_top_False.h5',
        'md5': '70c1759765a44a242ea3e8fe2161eb15',
    },
    # resnext101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnext101_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnext101_inp_channel_3_tch_0_top_False.h5',
        'md5': 'ca22ebb2456f2d05deddf0ab347e7782',
    },
    # vgg16
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/vgg16_inp_channel_3_tch_0_top_False.h5',
        'name': 'vgg16_inp_channel_3_tch_0_top_False.h5',
        'md5': '240d399c45ed038a5a7b026d750ceb2b',
    },
    # vgg19
    {
        'model': 'vgg19',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/vgg19_inp_channel_3_tch_0_top_False.h5',
        'name': 'vgg19_inp_channel_3_tch_0_top_False.h5',
        'md5': 'e081e19678fc76926a9f6ae93cd3d068',
    },
    # densenet121
    {
        'model': 'densenet121',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/densenet121_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet121_inp_channel_3_tch_0_top_False.h5',
        'md5': '1787a6780f62f338d481c460205443cd',
    },
    # densenet169
    {
        'model': 'densenet169',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/densenet169_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet169_inp_channel_3_tch_0_top_False.h5',
        'md5': '757a04f7ba7e473188bbbc6035f27193',
    },
    # densenet201
    {
        'model': 'densenet201',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/densenet201_inp_channel_3_tch_0_top_False.h5',
        'name': 'densenet201_inp_channel_3_tch_0_top_False.h5',
        'md5': '3ff42d034c530cb48f53576bfc68d06d',
    },
    # mobilenet
    {
        'model': 'mobilenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/mobilenet_inp_channel_3_tch_0_top_False.h5',
        'name': 'mobilenet_inp_channel_3_tch_0_top_False.h5',
        'md5': '88915a4dcfde7a1f223f6bc172078c7b',
    },
    # mobilenetv2
    {
        'model': 'mobilenetv2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/mobilenetv2_inp_channel_3_tch_0_top_False.h5',
        'name': 'mobilenetv2_inp_channel_3_tch_0_top_False.h5',
        'md5': '71cbf6db03cbf59930abc61e727cfa37',
    },
    # efficientnetb0
    {
        'model': 'efficientnetb0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb0_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb0_inp_channel_3_tch_0_top_False.h5',
        'md5': '2b85952f4972b4ec0d0733de35bab813',
    },
    # efficientnetb1
    {
        'model': 'efficientnetb1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb1_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb1_inp_channel_3_tch_0_top_False.h5',
        'md5': 'f7b33d083ef183104de8b6920cc88f52',
    },
    # efficientnetb2
    {
        'model': 'efficientnetb2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb2_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb2_inp_channel_3_tch_0_top_False.h5',
        'md5': '284ae21960a89a07e5afb388a3e6e7c3',
    },
    # efficientnetb3
    {
        'model': 'efficientnetb3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb3_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb3_inp_channel_3_tch_0_top_False.h5',
        'md5': 'a719d1bfe1d926935d0b51ca84f46c3b',
    },
    # efficientnetb4
    {
        'model': 'efficientnetb4',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb4_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb4_inp_channel_3_tch_0_top_False.h5',
        'md5': '0f0c45c0843a83997b83ac9b35a772b2',
    },
    # efficientnetb5
    {
        'model': 'efficientnetb5',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb5_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb5_inp_channel_3_tch_0_top_False.h5',
        'md5': '208f1279429ec2e26f60a362610cac38',
    },
    # efficientnetb6
    {
        'model': 'efficientnetb6',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb6_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb6_inp_channel_3_tch_0_top_False.h5',
        'md5': '8fc38ae4115ba407173176b3e92bc729',
    },
    # efficientnetb7
    {
        'model': 'efficientnetb7',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetb7_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetb7_inp_channel_3_tch_0_top_False.h5',
        'md5': '8d206d8a82937de28033f5a244ded863',
    },
    # efficientnetv2-b0
    {
        'model': 'efficientnetv2-b0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-b0_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-b0_inp_channel_3_tch_0_top_False.h5',
        'md5': '316a75d82fe65f5501110e21960edc5b',
    },
    # efficientnetv2-b1
    {
        'model': 'efficientnetv2-b1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-b1_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-b1_inp_channel_3_tch_0_top_False.h5',
        'md5': '632376206850d093990df8509ea3da2c',
    },
    # efficientnetv2-b2
    {
        'model': 'efficientnetv2-b2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-b2_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-b2_inp_channel_3_tch_0_top_False.h5',
        'md5': '504b436f55929c7fe0f243f88cd12d76',
    },
    # efficientnetv2-b3
    {
        'model': 'efficientnetv2-b3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-b3_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-b3_inp_channel_3_tch_0_top_False.h5',
        'md5': '7f222122a5c15ad15412f00f01c9391b',
    },
    # efficientnetv2-s
    {
        'model': 'efficientnetv2-s',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-s_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-s_inp_channel_3_tch_0_top_False.h5',
        'md5': '9cf1f3662cca4b25b505be1fade18df3',
    },
    # efficientnetv2-m
    {
        'model': 'efficientnetv2-m',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-m_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-m_inp_channel_3_tch_0_top_False.h5',
        'md5': '89f2334c79ea7804f5ec92855d6b1456',
    },
    # efficientnetv2-l
    {
        'model': 'efficientnetv2-l',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/efficientnetv2-l_inp_channel_3_tch_0_top_False.h5',
        'name': 'efficientnetv2-l_inp_channel_3_tch_0_top_False.h5',
        'md5': '985b5fa4fa572a02c60a01605e112192',
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
        'include_top': True,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/mobilenetv2_inp_channel_3_tch_0_top_True.h5',
        'name': 'mobilenetv2_inp_channel_3_tch_0_top_True.h5',
        'md5': 'edf2cea91f5e343cd66c489c9811bdf0',
    },
]
