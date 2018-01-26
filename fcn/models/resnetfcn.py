from __future__ import print_function
import collections
import os

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable

import chainer.links as L
import chainer.functions as F
import numpy as np
from .. import data
from .. import initializers
from chainer.links.model.vision.resnet import ResNetLayers


class ResNetLayersFCN32(link.Chain):


    def __init__(self, pretrained_model, n_layers, n_class, class_weight=None):
        super(ResNetLayersFCN32, self).__init__()
        self.n_class = n_class
        if class_weight is not None:
            assert class_weight.shape == (self.n_class,)
            self.class_weight = class_weight
        else:
            self.class_weight = None

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}

        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        kwargs2 = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
            }

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope(): #in the comments are the sizes (of default images of 224x224) AFTER the cooresponding layer
            self.conv1 = Convolution2D(3, 64, 7, 2, 3, **kwargs)                #112x112
            self.bn1 = BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, **kwargs)       #56x56
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, **kwargs)     #28x28
            self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2, **kwargs)    #14x14
            self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2, **kwargs)   #7x7
            #self.fc6 = Linear(2048, 1000)
            self.score_fr = L.Convolution2D(2048, n_class, 1, 1, 0, **kwargs2)
            self.upscore = L.Deconvolution2D(n_class, n_class, 64, 32, 0, nobias=True, initialW=initializers.UpsamplingDeconvWeight()) #224x224

        if pretrained_model and pretrained_model.endswith('.caffemodel'):  #default resnet model
            originalresnet = ResNetLayers(pretrained_model, n_layers)
            if n_layers == 50:
                _transfer_resnet50(originalresnet, self)
            elif n_layers == 101:
                _transfer_resnet101(originalresnet, self)
            elif n_layers == 152:
                _transfer_resnet152(originalresnet, self)
            else:
                raise ValueError('The n_layers argument should be either 50, 101,'
                                 ' or 152, but {} was given.'.format(n_layers))

        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            #('pool5', [_global_average_pooling_2d]),  #not for us...
            #('fc6', [self.fc6]),
            ('score_fr', [self.score_fr]),
            ('upscore', [self.upscore]),
            #('prob', [softmax]), #the other fcn doesn't have this, as there its in the sofmax crossentropy
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())


    def __call__(self, x, t=None, layers=None):
        #layers gets ignorted in this implementation
        if layers is not None:
            print("WARNING: passes value for layer, but dealing with this is not implemented yet!")

        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if key == 'upscore':  #last layer
                lastlayerout = h

        #lastlayerout = lastlayerout[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]

        #minumum increase in size: 30  maximum increase in size: 61   (increase in size depends on original image size, if its a (mulitple of 32)+2 the increase is 30)
        #   in  -> out
        #   516 -> 576
        #   515 -> 576
        #   514 -> 544
        #   513 -> 544
        #   512 -> 544
        #   511 -> 544
        #   510 -> 544
        #      ...
        #   483 -> 544
        #   482 -> 512
        #   481 -> 512
        #      ...
        #   451 -> 512
        #   450 -> 480
        #      ...


        #print("Just FYI: In dimensions: {}  Out dimensions:  {}".format(x.data.shape, lastlayerout.data.shape))   

        if lastlayerout.data.shape[2] >= x.data.shape[2] and lastlayerout.data.shape[3] >= x.data.shape[3]:
            xoffset = 15 #15=30/2  #int((lastlayerout.data.shape[2] - x.data.shape[2]) / 2)
            yoffset = 15 #15=30/2  #int((lastlayerout.data.shape[3] - x.data.shape[3]) / 2)
            lastlayerout = lastlayerout[:, :, xoffset:xoffset + x.data.shape[2], yoffset:yoffset + x.data.shape[3]]
        else:
            print("Output is smaller than input. This should not happen. Setting output to all 0 in the correct shape to avoid a crash. If this message ever shows up, we should implement a solution. In dimensions: {}  Out dimensions:  {}".format(x.data.shape, lastlayerout.data.shape))
            lastlayerout = np.zeros(x.data.shape)

        if lastlayerout.data.shape != x.data.shape:
            print("Ok, this should not be happeing, even after cropping we have a dimension mismatch. In dimensions: {}  Out dimensions:  {}".format(x.data.shape, lastlayerout.data.shape))   


        self.score = lastlayerout

        if t is None:
            assert not chainer.config.train
            return

        loss = F.softmax_cross_entropy(self.score, t, normalize=False, class_weight=self.class_weight)
        if np.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')
        return loss


    def predict(self, imgs):
        lbls = []
        for img in imgs:
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                x = self.xp.asarray(img[None])
                self.__call__(x)
                lbl = chainer.functions.argmax(self.score, axis=1)
            lbl = chainer.cuda.to_cpu(lbl.array[0])
            lbls.append(lbl)
        return lbls



class ResNet50LayersFCN32(ResNetLayersFCN32):

    def __init__(self, pretrained_model='auto', n_class=3, class_weight=None):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-50-model.caffemodel'
        super(ResNet50LayersFCN32, self).__init__(pretrained_model, 50, n_class=n_class, class_weight=class_weight)


class ResNet101LayersFCN32(ResNetLayersFCN32):

    def __init__(self, pretrained_model='auto', n_class=3, class_weight=None):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-101-model.caffemodel'
        super(ResNet101LayersFCN32, self).__init__(pretrained_model, 101, n_class=n_class, class_weight=class_weight)


class ResNet152LayersFCN32(ResNetLayersFCN32):

    def __init__(self, pretrained_model='auto', n_class=3, class_weight=None):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-152-model.caffemodel'
        super(ResNet152LayersFCN32, self).__init__(pretrained_model, 152, n_class=n_class, class_weight=class_weight)


def prepare(image, size=(224, 224)):
    """Converts the given image to the numpy array for ResNets.

    Note that you have to call this method before ``__call__``
    because the pre-trained resnet model requires to resize the given
    image, covert the RGB to the BGR, subtract the mean,
    and permute the dimensions before calling.

    Args:
        image (PIL.Image or numpy.ndarray): Input image.
            If an input is ``numpy.ndarray``, its shape must be
            ``(height, width)``, ``(height, width, channels)``,
            or ``(channels, height, width)``, and
            the order of the channels must be RGB.
        size (pair of ints): Size of converted images.
            If ``None``, the given image is not resized.

    Returns:
        numpy.ndarray: The converted output array.

    """

    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
    if isinstance(image, numpy.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = numpy.asarray(image, dtype=numpy.float32)
    image = image[:, :, ::-1]
    # NOTE: in the original paper they subtract a fixed mean image,
    #       however, in order to support arbitrary size we instead use the
    #       mean pixel (rather than mean image) as with VGG team. The mean
    #       value used in ResNet is slightly different from that of VGG16.
    image -= numpy.array(
        [103.063,  115.903,  123.152], dtype=numpy.float32)
    image = image.transpose((2, 0, 1))
    return image


class BuildingBlock(link.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(link.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(out_channels)
            self.conv4 = Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = relu(self.bn1(self.conv1(x)))
        h1 = relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return relu(h1 + h2)


class BottleneckB(link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(in_channels)

    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return relu(h + x)


#def _transfer_components(src, dst_conv, dst_bn, bname, cname):
 #   src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
 #   src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
 #   src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
 #   dst_conv.W.data[:] = src_conv.W.data
 #   dst_bn.avg_mean[:] = src_bn.avg_mean
 #   dst_bn.avg_var[:] = src_bn.avg_var
 #   dst_bn.gamma.data[:] = src_scale.gamma.data
 #   dst_bn.beta.data[:] = src_scale.beta.data

def _transfer_components(src_conv, src_bn, dst_conv, dst_bn):
    dst_conv.W.data[:]   = src_conv.W.data
    dst_bn.avg_mean[:]   = src_bn.avg_mean
    dst_bn.avg_var[:]    = src_bn.avg_var
    dst_bn.gamma.data[:] = src_bn.gamma.data
    dst_bn.beta.data[:]  = src_bn.beta.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src.conv1, src.bn1, dst.conv1, dst.bn1)
    _transfer_components(src.conv2, src.bn2, dst.conv2, dst.bn2)
    _transfer_components(src.conv3, src.bn3, dst.conv3, dst.bn3)
    _transfer_components(src.conv4, src.bn4, dst.conv4, dst.bn4)


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src.conv1, src.bn1, dst.conv1, dst.bn1)
    _transfer_components(src.conv2, src.bn2, dst.conv2, dst.bn2)
    _transfer_components(src.conv3, src.bn3, dst.conv3, dst.bn3)


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src.a, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        src_bottleneckB = getattr(src, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src_bottleneckB, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn1.avg_mean
    dst.bn1.avg_var[:] = src.bn1.avg_var
    dst.bn1.gamma.data[:] = src.bn1.gamma.data
    dst.bn1.beta.data[:] = src.bn1.beta.data

    _transfer_block(src.res2, dst.res2, ['2a', '2b', '2c'])   #these names are not used anymore, but we need to keep the array, cause its numer of elements is still in use
    _transfer_block(src.res3, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src.res4, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src.res5, dst.res5, ['5a', '5b', '5c'])

    #dst.fc6.W.data[:] = src.fc1000.W.data
    #dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    #dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn1.avg_mean
    dst.bn1.avg_var[:] = src.bn1.avg_var
    dst.bn1.gamma.data[:] = src.bn1.gamma.data
    dst.bn1.beta.data[:] = src.bn1.beta.data

    _transfer_block(src.res2, dst.res2, ['2a', '2b', '2c']) #these names are not used anymore, but we need to keep the array, cause its numer of elements is still in use
    _transfer_block(src.res3, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src.res4, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 23)])
    _transfer_block(src.res5, dst.res5, ['5a', '5b', '5c'])

    #dst.fc6.W.data[:] = src.fc1000.W.data
    #dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet152(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    #dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn1.avg_mean
    dst.bn1.avg_var[:] = src.bn1.avg_var
    dst.bn1.gamma.data[:] = src.bn1.gamma.data
    dst.bn1.beta.data[:] = src.bn1.beta.data

    _transfer_block(src.res2, dst.res2, ['2a', '2b', '2c'])  #these names are not used anymore, but we need to keep the array, cause its numer of elements is still in use
    _transfer_block(src.res3, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src.res4, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src.res5, dst.res5, ['5a', '5b', '5c'])

    #dst.fc6.W.data[:] = src.fc1000.W.data
    #dst.fc6.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model, n_layers):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))

    ResNetLayersFCN32.convert_caffemodel_to_npz(path_caffemodel, path_npz, n_layers)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(n_layers, name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model, n_layers),
        lambda path: npz.load_npz(path, model))
