"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union
import tempfile
import tifffile as tf
import os
from pathlib import Path
from .utils.vr_project import CreateProject

from napari.utils import progress
if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    """Writes a single image layer.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """

    path = Path(path)
    # implement your writer logic here ...
    project_name = path.parent.name
    project_path = path.parent.parent
    project = CreateProject(project_path, project_name)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # print('created temporary directory', tmpdirname)
        for i in progress( range(data.shape[0])):
            d = data[i]
            tf.imwrite(os.path.join(tmpdirname, f'image_{i}.tiff'), d)
        project.add_data(os.path.join(tmpdirname, 'image_0.tiff'))
    # return path to any file(s) that were successfully written
    return [path]


def write_multiple(path: str, data_layers: list[FullLayerData]) -> list[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """

    path = Path(path)
    # implement your writer logic here ...
    project_name = str(path.stem)
    project_path = str(path.parent)
    project = CreateProject(project_path, project_name)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # print('created temporary directory', tmpdirname)
        # print("number of layers: ",len(data_layers))
        for layer in data_layers:
            if layer[2] == 'image':
                image_data = layer[0]
                os.makedirs(os.path.join(tmpdirname, 'images'), exist_ok=True)
                for i in progress( range(image_data.shape[0]), desc='writing images'):
                    d = image_data[i]
                    tf.imwrite(os.path.join(tmpdirname, "images", f'image_{i}.tiff'), d)
                if os.path.exists(os.path.join(tmpdirname, "images", 'image_0.tiff')):
                    project.add_data(os.path.join(tmpdirname, "images", 'image_0.tiff'))
            elif layer[2] == 'labels':
                label_data = layer[0]
                os.makedirs(os.path.join(tmpdirname, 'labels'), exist_ok=True)
                for i in progress( range(label_data.shape[0]), desc='writing masks'):
                    m = label_data[i]
                    tf.imwrite(os.path.join(tmpdirname, "labels", f'labels_{i}.tiff'), m)      
                if os.path.exists(os.path.join(tmpdirname, "labels", 'labels_0.tiff')):
                    project.add_mask(os.path.join(tmpdirname, "labels", 'labels_0.tiff'))
    # return path to any file(s) that were successfully written
    return [path]
