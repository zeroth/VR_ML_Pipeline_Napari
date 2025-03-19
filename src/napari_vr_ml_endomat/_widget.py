from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget, Button
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import json
import os
import numpy as np
from .utils.image_analysis import filter_image, locate_blobs_single_channel, mask_blobs, get_cell_mask

from napari.utils import progress
if TYPE_CHECKING:
    import napari

class VRMLEndoMatSegmentation(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        
        func_list = ('filter_image', 'locate_blobs', 'mask_blobs')
    

        with open(os.path.join(os.path.abspath(__file__),'..','utils', 'endomat_parameters.json'), 'r') as f:
            self.parameters = json.load(f)
            for func_name in func_list:
                if func_name not in self.parameters:
                    self.parameters[func_name] = dict()
                print(f"{func_name} : ", self.parameters[func_name])
       
        self.name = "EndoMat"
        self._endomat_label = create_widget(label="EndoMat", annotation=str, widget_type="Label")
        self._endomat_label.value = "Segmentation"

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        
        self._threshold_slider = create_widget(
            label="Threshold", annotation=float, widget_type="FloatSlider"
        )
        self._max_sigma_slider = create_widget(
            label="Max Sigma", annotation=float, widget_type="FloatSlider"
        )

        self._min_sigma_slider = create_widget(
            label="Min Sigma", annotation=float, widget_type="FloatSlider"
        )

        self._no_sigma_slider = create_widget(
            label="Number of Sigma", annotation=float, widget_type="Slider"
        )

        self._overlap_slider = create_widget(
            label="Overlap", annotation=float, widget_type="FloatSlider"
        )

        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        self._threshold_slider.value = 0.1

        self._max_sigma_slider.min = 0
        self._max_sigma_slider.max = 10
        self._max_sigma_slider.value = 2

        self._min_sigma_slider.min = 0
        self._min_sigma_slider.max = 10
        self._min_sigma_slider.value = 1

        self._no_sigma_slider.min = 0
        self._no_sigma_slider.max = 10
        self._no_sigma_slider.value = 10

        self._overlap_slider.min = 0
        self._overlap_slider.max = 1
        self._overlap_slider.value = 0.5

        self._run_btn = Button(text="Segment")
        
        self._run_btn.changed.connect(self._get_mask)
        
        self.extend(
            [
                self._endomat_label,
                self._image_layer_combo,
                self._threshold_slider,
                self._min_sigma_slider,
                self._max_sigma_slider,
                self._no_sigma_slider,
                self._overlap_slider,
                self._run_btn,
            ]
        )

    def _get_mask(self):
        """
        filter_image :  {
            'kernel_size': [10, 20, 20], 
            'footprint': [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
            }
        locate_blobs :  {'min_sigma': [1.0, 1.0, 1.0], 'max_sigma': [2.0, 2.0, 2.0], 'num_sigma': 10, 'overlap': 0.5, 'threshold': 0.1, 'exclude_border': False, 'log_scale': True}
        mask_blobs :  {}
        """
        _locate_blobs = self.parameters['locate_blobs']
        _locate_blobs['min_sigma'] = [self._min_sigma_slider.value, self._min_sigma_slider.value, self._min_sigma_slider.value]
        _locate_blobs['max_sigma'] = [self._max_sigma_slider.value, self._max_sigma_slider.value, self._max_sigma_slider.value]
        _locate_blobs['num_sigma'] = self._no_sigma_slider.value
        _locate_blobs['overlap'] = self._overlap_slider.value
        _locate_blobs['threshold'] = self._threshold_slider.value
        
        image_layer = self._image_layer_combo.value
        _img_output = np.zeros(image_layer.data.shape, dtype=np.uint16)
        # print("_img_output.shape: ", _img_output.shape)
        for i in progress(range(image_layer.data.shape[0]), desc='Creating masks'):
            image = image_layer.data[i]
            _filter_image = filter_image(image, **self.parameters['filter_image'])
            _, _blobs = locate_blobs_single_channel(_filter_image, **_locate_blobs)
            _mask = mask_blobs(_blobs, image.shape, **self.parameters['mask_blobs'])
            # print("_mask.shape: ", _mask.shape)
            _img_output[i] = _mask
            # get_cell_mask(image, **self.parameters['get_cell_mask'])
        
        name = image_layer.name + "_mask"
        if 'mask' in self._viewer.layers:
            self._viewer.layers[name].data = _img_output
        else:
            self._viewer.add_labels(_img_output, name=name)

class VRMLEndoMatTracking(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        
        func_list = ('filter_image', 'locate_blobs', 'mask_blobs')
    

        with open(os.path.join(os.path.abspath(__file__),'..','utils', 'endomat_parameters.json'), 'r') as f:
            self.parameters = json.load(f)
            for func_name in func_list:
                if func_name not in self.parameters:
                    self.parameters[func_name] = dict()
                # print(f"{func_name} : ", self.parameters[func_name])
       
        self.name = "EndoMat"
        self._endomat_label = create_widget(label="EndoMat", annotation=str, widget_type="Label")
        self._endomat_label.value = "Tracking"

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._mask_layer_combo = create_widget(
            label="Mask", annotation="napari.layers.Labels"
        )
        
        self._search_range_slider = create_widget(
            label="Search Range", annotation=float, widget_type="FloatSlider"
        )
    
        self._memory_slider = create_widget(
            label="Memory", annotation=float, widget_type="Slider"
        )

        self._search_range_slider.min = 0
        self._search_range_slider.max = 100
        self._search_range_slider.value = 0.5

        

        self._memory_slider.min = 0
        self._memory_slider.max = self._mask_layer_combo.value.data.shape[0] - 1 if self._mask_layer_combo.value is not None else 1
        self._memory_slider.value = 0

        
        self._run_btn = Button(text="Track")
        
        self._run_btn.changed.connect(self._get_tracks)
        
        self.extend(
            [
                self._endomat_label,
                self._image_layer_combo,
                self._mask_layer_combo,
                self._search_range_slider,
                self._memory_slider,
                self._run_btn,
            ]
        )

    def _get_tracks(self):
        # do tracking related things here
        pass

class VRMLEndoMat(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._segmentation = VRMLEndoMatSegmentation(viewer)
        self._tracking = VRMLEndoMatTracking(viewer)
        self.labels = False

        self.extend(
            [
                self._segmentation,
                self._tracking,
            ]
        )




class TrainWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._sy_path = create_widget(
            label="syGlass Project", annotation="str", widget_type="FileEdit", options={"mode": "r", "filter": "syGlass Project (*.syg)"}
        )

        self._or_label = create_widget(label="or", annotation=str, widget_type="Label")

        self._tiff_image_path = create_widget(
            label="Tiff Image", annotation="str", widget_type="FileEdit", options={"mode": "d", "filter": "Tiff Image (*.tif)"}
        )

        self._tiff_label_path = create_widget(
            label="Tiff Label", annotation="str", widget_type="FileEdit", options={"mode": "d", "filter": "Tiff Label (*.tif)"}
        )
        
        self._train_btn = Button(text="Train")
        self._train_btn.changed.connect(self._train)
        self.extend(
            [
                self._sy_path,
                self._or_label,
                self._tiff_image_path,
                self._tiff_label_path,
                self._train_btn,
            ]
        )

    def _train(self):
        # do training related things here
        pass

class InferenceWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._sy_path = create_widget(
            label="syGlass Project", annotation="str", widget_type="FileEdit", options={"mode": "r", "filter": "syGlass Project (*.syg)"}
        )

        self._or_label = create_widget(label="or", annotation=str, widget_type="Label")

        self._tiff_image_path = create_widget(
            label="Tiff Image", annotation="str", widget_type="FileEdit", options={"mode": "d", "filter": "Tiff Image (*.tif)"}
        )

        self._inference_btn = Button(text="Inference")
        self._inference_btn.changed.connect(self._inference)
        self.extend(
            [
                self._sy_path,
                self._or_label,
                self._tiff_image_path,
                self._inference_btn,
            ]
        )

    def _inference(self):
        # do inference related things here
        pass