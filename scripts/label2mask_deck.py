# -*- coding: utf-8 -*-
import numpy as np
import xmltodict

from PIL import Image, ImageDraw


def get_xml():
    with open('0049.xml') as fd:
        doc = xmltodict.parse(fd.read())
        return doc


def remove_deleted_mask(doc):
    damage_list = []
    mask_object = doc['annotation']['object']
    for damage in mask_object:
        if damage['deleted'] == '0':
            damage_list.append(damage)

    return damage_list


def get_points(polygon):
    points = []
    for poly in polygon:
        for pt in poly['pt']:
            points.append((int(pt['x']), int(pt['y'])))

    return points


def get_mask(damage_list, img_shape, damage, color, final_mask):
    w, h = img_shape
    for dmg in damage_list:
        if dmg['name'] == damage:
            polygon = [dmg['polygon']]
            points = get_points(polygon)
            mask = np.zeros(img_shape, dtype=np.uint8)
            mask = Image.fromarray(mask)
            ImageDraw.Draw(mask).polygon(xy=points, outline=1, fill=1)
            mask = np.array(mask, dtype=np.uint8)
            color_mask = np.empty((w, h, 3), dtype=np.uint8)
            color_mask[:, :, 1] = color_mask[:, :, 0] = mask
            color_mask[:, :, 2] = 0
            final_mask[color_mask[:, :, 0] == 1] = color

    return final_mask


def paint_mask(damage_list, img_shape):
    white = [1, 1, 1]
    w, h = img_shape
    mask = np.zeros((w, h, 3), dtype=np.uint8)
    deck_mask = get_mask(
                                damage_list,
                                img_shape,
                                'deck',
                                white,
                                mask)

    return Image.fromarray(deck_mask * 255)


if __name__ == '__main__':
    doc = get_xml()
    height = int(doc['annotation']['imagesize']['nrows'])
    width = int(doc['annotation']['imagesize']['ncols'])
    damage_list = remove_deleted_mask(doc)
    result = paint_mask(damage_list, [height, width])
    result.save('deck_result.png')
