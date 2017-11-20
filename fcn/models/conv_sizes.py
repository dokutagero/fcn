# pip install chainer
from chainer.utils.conv import get_conv_outsize
from chainer.utils.conv import get_deconv_outsize


# name, ksize, stride, pad
layers = [
    ('conv1_1', 3, 1, 100),
    ('conv1_2', 3, 1, 1),
    ('pool1', 2, 2, 0),

    ('conv2_1', 3, 1, 1),
    ('conv2_2', 3, 1, 1),
    ('pool2', 2, 2, 0),

    ('conv3_1', 3, 1, 1),
    ('conv3_2', 3, 1, 1),
    ('conv3_3', 3, 1, 1),
    ('pool3', 2, 2, 0),

    ('conv4_1', 3, 1, 1),
    ('conv4_2', 3, 1, 1),
    ('conv4_3', 3, 1, 1),
    ('pool4', 2, 2, 0),

    ('conv5_1', 3, 1, 1),
    ('conv5_2', 3, 1, 1),
    ('conv5_3', 3, 1, 1),
    ('pool5', 2, 2, 0),

    ('fc6', 7, 1, 0),
    ('fc7', 1, 1, 0),

    ('score_fr', 1, 1, 0),
    ('upscore', 64, 32, 0),
]


def get_crop_pad(in_axis):
    in_axis_original = in_axis
    for name, ksize, stride, pad in layers:
        cover_all = name.startswith('pool')
        if name.startswith('up'):
            out_axis = get_deconv_outsize(in_axis, ksize, stride, pad)
        else:
            out_axis = get_conv_outsize(in_axis, ksize, stride, pad,
                                        cover_all=cover_all)
        in_axis = out_axis
    offset = (out_axis - in_axis_original) / 2.
    return offset

def main():
    offsets = []
    for in_hw in range(1, 2000):  # hw: height or width of input image
        offset = get_crop_pad(in_hw)
        offsets.append(offset)
    # print(offsets)
    print(min(offsets), max(offsets))
    #     in_hw_org = in_hw
    #     for name, ksize, stride, pad in layers:
    #         cover_all = name.startswith('pool')
    #         if name.startswith('up'):
    #             out_hw = get_deconv_outsize(in_hw, ksize, stride, pad)
    #         else:
    #             out_hw = get_conv_outsize(in_hw, ksize, stride, pad,
    #                                       cover_all=cover_all)
    #         in_hw = out_hw
    #     offset = (out_hw - in_hw_org) / 2.
    #     print(offset)
    #     offsets.append(offset)
    # print(min(offsets), max(offsets))  # (19.0, 34.5)


if "__main__" == __name__:
    main()
