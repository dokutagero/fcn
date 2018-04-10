# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import xmltodict

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def kilo(x, pos):
    'The two args are the value and tick position'
    return '{}k'.format(x*1e-3)


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
    formatter = FuncFormatter(kilo)

    delamination_fig = plt.figure(1)
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)
    # plt.hist(delamination_blob_size, bins=1000, orientation='horizontal')
    plt.hist(delamination_blob_size, bins=5000)
    plt.title('Histogram of Delamination Blob Size')

    rebar_exposure_fig = plt.figure(2)
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    ax = plt.gca()
    # ax.yaxis.set_major_formatter(formatter)
    plt.hist(rebar_exposure_blob_size, bins=2500)
    plt.title('Histogram of Rebar Exposure Blob Size')

    delamination_fig.savefig('delamination_blog_hist.png')
    rebar_exposure_fig.savefig('rebar_exposure_blog_hist.png')
    # plt.show()


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
    XML_FILE_PATH = '/Users/mrteera/workspace/blob_size/'
    annotators = ['bridge_annotations_ie1/', 'bridge_annotations_ie2/', 'bridge_annotations_ie3/']
    delamination_areas = []
    rebar_exposure_areas = []
    file_list = []
    for annotator in annotators:
        for deck in ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']:
            for xml_file in os.listdir(XML_FILE_PATH + annotator + deck):
                file_list.append(XML_FILE_PATH + annotator + deck + xml_file)
    # Remove duplicated XML files
    xml_file_list = []
    for file in file_list:
        xml_file_list.append('/'.join(file.split('/')[-2:]))

    final_xml_list = []
    for xml_file in sorted(xml_file_list):
        if xml_file_list.count(xml_file) == 3:
            final_xml_list.append(xml_file)
    final_xml_list = sorted(list(set(final_xml_list)))

    for xml_file in final_xml_list:
        # delamination_blob_sizes = []
        # rebar_exposure_blob_sizes = []
        for annotator in annotators:
            delamination_blob_size, rebar_exposure_blob_size = get_blob_sizes(XML_FILE_PATH + annotator + xml_file)
            # delamination_blob_sizes.append(delamination_blob_size)
            # rebar_exposure_blob_sizes.append(rebar_exposure_blob_size)

            if sum(rebar_exposure_blob_size) != 0:
                rebar_exposure_areas = rebar_exposure_areas + rebar_exposure_blob_size
            if sum(delamination_blob_size) != 0:
                delamination_areas = delamination_areas + delamination_blob_size

    # print(len(delamination_areas))
    # print(len(rebar_exposure_areas))
    delamination_areas = sorted(delamination_areas)
    delamination_areas = delamination_areas[int(len(delamination_areas) * .05) : int(len(delamination_areas) * .95)]
    rebar_exposure_areas = sorted(rebar_exposure_areas)
    rebar_exposure_areas = rebar_exposure_areas[int(len(rebar_exposure_areas) * .05) : int(len(rebar_exposure_areas) * .95)]
    # print('delamination len: ', len(delamination_areas))
    # print('rebar exposure len: ', len(rebar_exposure_areas))

    # print('=-=-=-=-=-=-=-=-=-=-=-=- DELAMINATION =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    # print(sorted(delamination_areas))
    # print('=-=-=-=-=-=-=-=-=-=-=- REBAR EXPOSURE =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    # print(sorted(rebar_exposure_areas))
    plot_hist(delamination_areas, rebar_exposure_areas)
