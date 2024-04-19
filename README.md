# DeepSpinePreprocessing
This repository contains the code used for the preprocessing phase of dendritic shaft and spines.

## Installation
Requires **Python 3.6.8 or later**.
1. Download the current project:
   > git clone https://github.com/vg-lab-dl/DeepSpinePreprocessing.git
2. Install dependencies (from the root directory of the project):
   > pip install -r requirements.txt.

## Usage
First, the following parameters must be specified in `config.yaml`:

- `spinesPath`: path to the directory with folders for every dendrite to process. Each folder contains all spines associated with the dendrite in **.obj** format.
- `dendriticShaftPath`: path to the directory with dendritic shafts in **.obj** format. 
- `rawPath`: path to the directory with raw images of dendrites from microscopy in **.tif** format.
- `metadataPath`: path to the directory with metadata of microscopy images in **.txt** format, following syntax:
   > ExtMax0={X max}<br>ExtMax1={Y max}<br>ExtMax2={Z max}<br>ExtMin0={X min}<br>ExtMin1={Y min}<br>ExtMin2={Z min}<br>X={stack width}<br>Y={stack height}<br>Z={stack depth}
                                                                                                       
   > ExtMaxX are the bounding box coordinates
- `outputPath`: directory where GT images will be saved in **.tif** format.
- `joinUnconnectedElements`: flag that specifies whether to attach components or not during the voxelization process

**Important!** Note that files in `dendriticShaftPath`, `rawPath` and `metadataPath` should be named the same way as the corresponding spines folder in `spinesPath`.

To run the preprocessing phase, use the following command:
> python main.py -cf=config.yaml

## Acknowledgments
The authors gratefully acknowledges the computer resources at Artemisa, funded by the European Union ERDF and Comunitat Valenciana as well as the technical support provided by the Instituto de Física Corpuscular, IFIC (CSIC-UV).

## License 
DeepSpinePreprocessing is distributed under a Dual License model, depending on its usage. For its non-commercial use, it is released under an open-source license (GPLv3). Please contact us if you are interested in commercial license.
