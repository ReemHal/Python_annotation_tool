#!/usr/bin/env python

import json
import math
import numpy
from PIL import Image


class AnnotationData:
    def __init__(self, labels, image_data):
        self._labels = labels
        self._image_shape = image_data.shape[:2]

        # Create a dictionary for each label
        # Each dictionary will map object numbers to their data
        self._label_sets = [dict() for i in range(len(self._labels))]
        # Create layer for each label (this will hold all objects on one layer)
        self._label_single = [
            numpy.zeros(self._image_shape, dtype=bool)
            for i in range(len(self._labels))
        ]

        raw_annotation = numpy.bitwise_or(numpy.bitwise_or(
            image_data[:, :, 0].astype(numpy.uint32),
            image_data[:, :, 1].astype(numpy.uint32) << 8),
            image_data[:, :, 2].astype(numpy.uint32) << 16)

        ones = numpy.ones_like(raw_annotation, dtype=numpy.uint8)
        zeros = numpy.zeros_like(raw_annotation, dtype=numpy.uint8)

        unique_combinations = numpy.unique(raw_annotation)
        for combination in unique_combinations:
            # Decode combination into set of labels and objects.
            layers = _rec_separate_layers(
                combined_layers=combination,
                times_combined=math.ceil(math.log(len(self._labels), 2))
            )
            for layer in layers:
                object_mask = numpy.where(raw_annotation == combination, ones, zeros)

                if layer[1] in self._label_sets[layer[0]]:
                    existing_mask = self._label_sets[layer[0]][layer[1]]
                    object_mask = numpy.logical_or(object_mask, existing_mask)

                # Add object to dict
                self._label_sets[layer[0]][layer[1]] = object_mask
                # Append object to layer
                self._label_single[layer[0]] = numpy.logical_or(
                    self._label_single[layer[0]], object_mask
                )

    def get_classes(self):
        """Returns a count of objects in each class
        
        :return: A dictionary eg. {'tomato': 5, 'leaf': 3, 'stem': 4}
        """
        dictionary = dict()
        for i, label in enumerate(self._labels):
            dictionary[label] = len(self._label_sets[i])
        return dictionary

    def get_mask(self, label_name, object_number=None):
        """Returns a boolean mask
        
        :param label_name: the label to mask eg. 'tomato'
        :param object_number: optional object number
        :return: An array the same size as the image with 1 representing the
            object and 0 elsewhere. If no object is specified, all objects
            with matching label will be masked as 1
        """
        try:
            label_index = self._labels.index(label_name)
        except ValueError:
            return None

        if object_number is None:
            return self._label_single[label_index]
        else:
            try:
                return self._label_sets[label_index][object_number]
            except KeyError:
                return None


def _rec_separate_layers(combined_layers, times_combined, counter=0):
    unpacked = _inverse_cantor_pair(combined_layers)
    layers = []
    if counter < times_combined:
        layers.extend(_rec_separate_layers(unpacked[0], times_combined, counter + 1))
        layers.extend(_rec_separate_layers(unpacked[1], times_combined, counter + 1))
    else:
        if unpacked[0] != 0:
            layers.append(unpacked)
    return layers


def _inverse_cantor_pair(z):
    w = math.floor((math.sqrt(8*z+1)-1)/2)
    t = (math.pow(w, 2) + w)/2
    y = int(z - t)
    x = int(w - y)
    return x, y


def read(annotation_path, json_path):
    """Parses a .json file to load annotation data from a .png file.
    
    :param annotation_path: The path to the annotation .png file
    :param json_path: The path to the .json file
    :return: an AnnotationData object containing the loaded data
    """

    try:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

    except IOError:
        print("Cannot find .json file")
        return None

    try:
        image_data = numpy.array(Image.open(annotation_path))
    except IOError:
        print("Cannot find annotation file")
        return None

    if "labels" not in json_data:
        print("Error parsing json file")
        return None

    # else
    return AnnotationData(json_data["labels"], image_data)
