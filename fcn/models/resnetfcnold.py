import chainer
import collections
from chainer.links.model.vision.resnet import ResNetLayers
import chainer.links as L
from chainer.functions.activation.relu import relu
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from .. import initializers

class ResNetLayersFCN(ResNetLayers):

    def __init__(self, pretrained_model, n_layers, n_class=2):
        super(ResNetLayersFCN, self).__init__(pretrained_model, n_layers)
        self.n_class = n_class

        #kwargs = {
        #    'initialW': chainer.initializers.Zero(),
        #    'initial_bias': chainer.initializers.Zero(),
        #}
        kwargs = {
            'initialW': chainer.initializers.Normal(),
            'initial_bias': chainer.initializers.Normal(),
        }

        del self.fc6

        with self.init_scope():
            self.score_fr = L.Convolution2D(2048, n_class, 1, 1, 0, **kwargs)
            self.upscore = L.Deconvolution2D(n_class, n_class, 64, 32, 0, nobias=True, initialW=initializers.UpsamplingDeconvWeight()) #224x224


    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),                             #112x112
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),         #56x56
            ('res2', [self.res2]),  # STRIDE FUCKING 1                           #56x56
            ('res3', [self.res3]),                                               #28x28
            ('res4', [self.res4]),                                               #14x14
            ('res5', [self.res5]),                                               #7x7
            #('pool5', [_global_average_pooling_2d]),  #not for us...
            #('fc6', [self.fc6]),
            ('score_fr', [self.score_fr]),
            ('upscore', [self.upscore]),
            #('prob', [softmax]), #the other fcn doesn't have this, as there its in the sofmax crossentropy
        ])

    def __call__(self, x, layers=None, **kwargs):
        #layers gets ignorted in this implementation
        if layers is not None:
            print("WARNING: passes value for layer, but dealing with this is not implemented yet!")

        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if key == 'upscore':  #last layer
                lastlayerout = h

        lastlayerout = lastlayerout[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        self.score = lastlayerout
        return self.score

    def predict(self, images, oversample=True):
        #doesn"t need to be reimplementerd for our purpose, so will remain broken
        assert False




    def preparePILimage(image, size=(224, 224)):
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
        #if size:
        #    image = image.resize(size)
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
