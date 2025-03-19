# author: Abhishek Patil <abhishek@zeroth.me>
# This file contains the class VRProject which is used to read and write data to a VR project
# The class has methods to read data and mask from the project and add data and mask to the project

import argparse
from syglass import pyglass
import syglass as sy
import numpy as np
import os
import time
from pathlib import Path


class VRProject():
    def __init__(self, project):
        self.project = project
        self.impl = project.impl

        self.resolution_map = self.project.get_resolution_map()
        self.mask_raster = pyglass.MaskOctreeRasterExtractor(None)
        self.max_resolution_level = len(self.resolution_map) - 1
        self.frame_count = self.project.get_timepoint_count()
        self.dimensions = self.project.get_size(self.max_resolution_level)
        self.voxel_dimensions = self.project.get_voxel_dimensions()

    def has_mask(self):
        return Path(self.get_project_path()).with_suffix('.syk').exists()
    
    def read_timepoint(self, timepoint):
        block = self.project.get_custom_block(timepoint, self.max_resolution_level, np.array([0, 0, 0]), self.dimensions)
        return np.copy(np.squeeze(block.data))

    def read_mask_timepoint(self, timepoint):
        vec3_dimensions = pyglass.vec3(float(self.dimensions[2]), float(self.dimensions[1]), float(self.dimensions[0]))
        vec3_offset = pyglass.vec3(float(0), float(0), float(0))
        raster = self.mask_raster.GetCustomBlock(self.impl, timepoint, self.max_resolution_level, vec3_offset, vec3_dimensions)
        block = sy.Block(pyglass.GetRasterAsNumpyArray(raster), np.array([0, 0, 0]))
        
        return np.copy(np.squeeze(block.data))
    
    def get_project_path(self):
        return self.project.get_path_to_syg_file().string()
    
    def add_data(self, tiff_file_path):
        dd = pyglass.DirectoryDescription()
        dd.InspectByReferenceFile(tiff_file_path)
        dataProvider = pyglass.OpenTIFFs(dd.GetFileList(), True)

        includedChannels = pyglass.IntList(range(dataProvider.GetChannelsCount()))
        dataProvider.SetIncludedChannels(includedChannels)

        cd = pyglass.ConversionDriver()
        cd.SetInput(dataProvider)
        cd.SetOutput(self.impl)
        cd.StartAsynchronous()

        while cd.GetPercentage() != 100:
            print(cd.GetPercentage())
            time.sleep(1)
        print(f"Project : {self.project.get_name()} data added")

    def add_mask(self, mask_file_path):
        _syk_path = Path(self.get_project_path()).with_suffix('.syk')
        p = pyglass.CreateProject(pyglass.path(str(_syk_path.parent)), _syk_path.name,  True)
        dd = pyglass.DirectoryDescription()
        dd.InspectByReferenceFile(mask_file_path)
        dataProvider = pyglass.OpenTIFFs(dd.GetFileList(), True)

        includedChannels = pyglass.IntList(range(dataProvider.GetChannelsCount()))
        dataProvider.SetIncludedChannels(includedChannels)

        cd = pyglass.ConversionDriver(True)
        cd.SetInput(dataProvider)
        cd.SetOutput(p)
        cd.StartAsynchronous()

        while cd.GetPercentage() != 100:
            print(cd.GetPercentage())
            time.sleep(1)
        print(f"Project : {self.project.get_name()} mask added")


def CreateProject(project_path, project_name):
    p = pyglass.CreateProject(pyglass.path(project_path), project_name)
    project = sy.Project(p)
    return VRProject(project)

def OpenProject(project_path, project_name):
    project = sy.get_project(os.path.join(
        project_path, project_name, f"{project_name}.syg"))
    return VRProject(project)

def GetProject(project_path, project_name):
    if os.path.exists(os.path.join(project_path, project_name, f"{project_name}.syg")):
        return OpenProject(project_path, project_name)
    else:
        return CreateProject(project_path, project_name)


def main():
    parser = argparse.ArgumentParser(description="Read a VR project")
    parser.add_argument("project_path", help="Path to the project")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("-n", "--new", action="store_true", help="Create a new project", default=False)

    args = parser.parse_args()
    project = OpenProject(args.project_path, args.project_name)
    print(project.project.get_name())
    print(project.get_project_path())

if __name__ == "__main__":
    main()