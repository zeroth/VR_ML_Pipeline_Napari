"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from .utils.vr_project import OpenProject
import dask.array as da
from dask import delayed
from pathlib import Path


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".syg"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Takes a path to .syg file  and return a list of LayerData tuples.
    Parameters
    ----------
    path : str Path to .syg file.

    Returns
    -------
    layer_data : list of tuples of image and/or label data
    """

    path = path[0] if isinstance(path, list) else path
    path = Path(path)
    project_name = path.parent.name
    project_path = path.parent.parent
    project = OpenProject(project_path, project_name)

    lazy_reader_timepoint = delayed(project.read_timepoint)

    sample = project.read_timepoint(0)
    sample_dtype = sample.dtype

    arrays = [lazy_reader_timepoint(index) for index in range(project.frame_count)]
    
    data_array = [
        da.from_delayed(array, shape=project.dimensions, dtype=sample_dtype)
        for array in arrays
    ]
    contrast = [sample.min(), sample.max()]
    data = da.stack(data_array, axis=0)
    add_kwargs = {
        'contrast_limits':contrast, 
        'multiscale':False,
        'scale':project.voxel_dimensions,
        "name": project_name,
        }

    image_layer = (data, add_kwargs, "image") # image layer


    # mask layer
    if not project.has_mask():
        return [image_layer]
    
    lazy_maks_reader_timepoint = delayed(project.read_mask_timepoint)
    mask_arrays = [lazy_maks_reader_timepoint(index) for index in range(project.frame_count)]
    
    data_mask_array = [
        da.from_delayed(array, shape=project.dimensions, dtype=np.uint16)
        for array in mask_arrays
    ]

    data_mask = da.stack(data_mask_array, axis=0)
    
    add_mask_kwargs = {
        'multiscale':False,
        'scale':project.voxel_dimensions,
        "name": f"{project_name}_mask",
        }
    mask_layer = (data_mask, add_mask_kwargs, "labels") # mask layer
    return [image_layer, mask_layer]
