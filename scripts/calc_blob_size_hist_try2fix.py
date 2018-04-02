# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import numpy as np
import os
import xmltodict

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

XML_FILE_PATH = '/root/bridge_dataset/bridge_masks_total/'

def get_xml(filename):
    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())
        return doc


def remove_deleted_mask(doc):
    damage_list = []
    mask_object = doc['annotation']['object']
    if not isinstance(mask_object, list):
        mask_object = [mask_object]
    for damage in mask_object:
        if damage['deleted'] == '0':
            damage_list.append(damage)

    return damage_list


# https://stackoverflow.com/a/24468019/1971997
def PolygonArea(damage_pt):
    corners = []
    for pt in damage_pt:
        corners.append((int(pt['x']), int(pt['y'])))
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return [area, corners]


def plot_hist(delamination_blob_size, rebar_exposure_blob_size):
    plt.figure(1)
    plt.hist(delamination_blob_size, bins=2500, orientation='horizontal')
    plt.title('Histogram of delamination blob size')
    plt.figure(2)
    plt.hist(rebar_exposure_blob_size, bins=2500, orientation='horizontal')
    plt.title('Histogram of rebar_exposure blob size')
    plt.show()


def get_blob_sizes(xml_file):
    doc = get_xml(xml_file)
    damage_list = remove_deleted_mask(doc)
    delamination_blob_size = []
    rebar_exposure_blob_size = []

    for damage in damage_list:
        delamination_corners = []
        rebar_exposure_corners = []
        if damage['name'] == 'delamination':
            [blob_size, delamination_corners] = PolygonArea(damage['polygon']['pt'])
            delamination_poly = Polygon(delamination_corners)
            for damage in damage_list:
                if damage['name'] == 'rebar_exposure':
                    has_rebar_in_delamination = []
                    for pt in damage['polygon']['pt']:
                        point = Point(int(pt['x']), int(pt['y']))
                        has_rebar_in_delamination.append(point.within(delamination_poly))
                    if all(has_rebar_in_delamination) == True:
                        blob_size = blob_size - PolygonArea(damage['polygon']['pt'])[0]
                    elif all(has_rebar_in_delamination) == False:
                        pass
                    else:
                        print('Something wrong with rebar_exposure')
            delamination_blob_size.append(blob_size)
        elif damage['name'] == 'rebar_exposure':
            blob_size = PolygonArea(damage['polygon']['pt'])[0]
            rebar_exposure_blob_size.append(blob_size)
    return delamination_blob_size, rebar_exposure_blob_size


if __name__ == '__main__':
    delamination_areas = []
    rebar_exposure_areas = []
    for deck in ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']:
        for xml_file in os.listdir(XML_FILE_PATH + deck):
            delamination_blob_size, rebar_exposure_blob_size = get_blob_sizes(XML_FILE_PATH + deck + xml_file)
            delamination_areas = delamination_areas + delamination_blob_size
            rebar_exposure_areas = rebar_exposure_areas + rebar_exposure_blob_size
    print('=-=-=-=-=-=-=-=-=-=-=-=- DELAMINATION =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print(sorted(delamination_areas))
    print('=-=-=-=-=-=-=-=-=-=-=- REBAR EXPOSURE =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print(sorted(rebar_exposure_areas))
    # plot_hist(delamination_areas, rebar_exposure_areas)
