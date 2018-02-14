#! /usr/bin/env python

import argparse
import io
import os
import shutil

import h5py
import numpy as np
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageFont, ImageDraw

argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to overfit on a single image.')


def draw_figure(draw, figure, x, y, w, h, fill):
    if figure == 'S':
        draw_rectangle(draw, x, y, w, h, fill)
    elif figure == 'C':
        draw_ellipse(draw, x, y, w, h, fill)
    elif figure == 'T':
        draw_t(draw, x, y, w, h, fill)


def draw_ellipse(draw, x, y, w, h, fill):
    draw.ellipse(((x, y), (x + w, y + h)), fill=fill)


def draw_rectangle(draw, x, y, w, h, fill):
    draw.rectangle(((x, y), (x + w, y + h)), fill=fill)


def draw_t(draw, x, y, w, h, fill):
    draw.polygon(((x, y + h), ((x + (w / 2)), y), (x + w, y + h)), fill=fill)


def draw_outline(draw, x, y, w, h):
    draw.rectangle(((x, y), (x + w, y + h)), outline="red")


# Find new position for figure (w, h)
def find_pos(figures_description, figure_type, w, h):
    image_padding = 10
    start_x = image_padding
    start_y = image_padding
    end_x = (512 - w - image_padding)
    end_y = (512 - h - image_padding)

    fill = True
    try_count = 0
    while fill:
        x = random.randint(start_x, end_x)
        y = random.randint(start_y, end_y)

        fill = is_fill(figures_description, x, y, x + w, y + h)
        try_count = try_count + 1

        if fill:
            if try_count > 100:
                print("Skipped!")
                return None
        else:
            return x, y


# Return true if place (x, y, w, h) is filled
def is_fill(figures_description, x, y, x_max, y_max):
    for exist_figure in figures_description:
        result = is_intersect(x - 5, y - 5, x_max + 5, y_max + 5, exist_figure[1], exist_figure[2], exist_figure[3], exist_figure[4])
        result2 = is_intersect(exist_figure[1], exist_figure[2], exist_figure[3], exist_figure[4], x - 5, y - 5, x_max + 5, y_max + 5)
        if result or result2:
            return True

    return False


def is_intersect(x1, y1, x1_max, y1_max, x2, y2, x2_max, y2_max):

    if abs(x1 - x2) < 10 or abs(y2 - y1) < 10:
        return True

    if x1 < x2 and y1 < y2 and x1_max > x2 and y1_max > y2:
        return True

    if x1 < x2_max and y1 < y2_max and x1_max > x2_max and y1_max > y2_max:
        return True

    if x1_max > x2_max and y1 < y2 and x1 < x2_max and y1_max > y2:
        return True

    if x1 < x2_max and y1_max > y2 and x1 > x2 and y1_max < y2_max:
        return True

    return False


# Prepare a directory.
# If a directory not found - create new directory
# else - clear the directory
def prepare_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        clear_dir(path)


# Remove all files from directory in 1 level only
# If directory not found - return
def clear_dir(path):
    if not os.path.exists(path):
        return
    if not os.path.isdir(path):
        return

    files = os.listdir(path)

    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isfile(full_path):
            os.remove(full_path)


def create_object_elem(desc, figures):
    _object = ET.Element("object")
    name = ET.Element("name")
    name.text = figures[desc[0]]
    _object.append(name)
    pose = ET.Element("pose")
    pose.text = "Unspecified"
    truncated = ET.Element("truncated")
    truncated.text = "0"
    difficult = ET.Element("difficult")
    difficult.text = "0"
    _object.append(pose)
    _object.append(truncated)
    _object.append(difficult)
    bndbox = ET.Element("bndbox")
    xmin = ET.Element("xmin")
    xmin.text = str(desc[1])
    bndbox.append(xmin)
    ymin = ET.Element("ymin")
    ymin.text = str(desc[2])
    bndbox.append(ymin)
    xmax = ET.Element("xmax")
    xmax.text = str(desc[3])
    bndbox.append(xmax)
    ymax = ET.Element("ymax")
    ymax.text = str(desc[4])
    bndbox.append(ymax)
    _object.append(bndbox)
    return _object


def _main(args):
    print('Test generate images...')
    print('======================================================')

    # Output directory  
    img_output_dir = "gen-output/images"
    annotation_output_dir = "gen-output/annotations"
    data_file = "gen-output/example.xml"

    # Image size
    image_width = 512
    image_height = 512
    image_size = (image_width, image_height)
    image_backgound = (194, 255, 196)

    # image = Image.new('RGB', (image_width, image_height), (255, 0, 0))
    # draw = ImageDraw.Draw(image)

    # draw.rectangle(((0,0),(10,10)), fill="black", outline = "blue")
    # dr.rectangle(((20,20),(70,70)), fill="black")

    image_count = 70

    figures = ['T', 'S', 'C']

    # Prepare output directory 
    prepare_dir(img_output_dir)
    prepare_dir(annotation_output_dir)

    for image_index in range(1, image_count + 1):

        figure_color = (207, 95, 255)

        image = Image.new('RGB', image_size, image_backgound)
        draw = ImageDraw.Draw(image)

        # Random figure count
        figure_count = random.randint(1, 9)

        print("Image [" + str(image_index) + "], Figures  [" + str(figure_count) + "]")

        # print(boxes)
        figures_description = []
        boxes = []
        for figure_index in range(1, figure_count + 1):

            # Figure size
            rand_size = random.randint(40, 300)
            figure_width = rand_size
            figure_height = rand_size

            # Random figure type
            figure_type = random.randint(0, 2)
            figure = figures[figure_type]

            # Random figure position
            coord = find_pos(figures_description, figure_type, figure_width, figure_height)
            if coord is not None:
                x = coord[0]
                y = coord[1]
                figures_description.append([figure_type, x - 5, y - 5, x + figure_width + 5, y + figure_height + 5])
                # Draw figure
                draw_figure(draw, figure, x, y, figure_width, figure_height, figure_color)

        tree = ET.ElementTree(file=data_file)
        file_name = "image_" + str(image_index) + ".jpg"
        image.save(img_output_dir + "/" + file_name)

        # Create xml
        tree.find("filename").text = file_name
        annotation = tree.getroot()

        for desc in figures_description:
            annotation.append(create_object_elem(desc, figures))

        tree.write(annotation_output_dir + "/image_" + str(image_index) + ".xml")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
