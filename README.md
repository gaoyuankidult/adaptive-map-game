This repo is used to store all the packages need to automaticly process a tobii folder.

### Getting Started
To get start, please first clone the repo to a local repository.

`git clone https://github.com/gaoyuankidult/adaptive-map-game.git`

#### Store your data in the ./tobii_file folder

Please menualy copy all the content in the tobii folder that you want to analyze to the `./tobii_file` file.

First, we need to use `tobii_preprocessing.py` script in the folder glassesCalibration.

To do this, we have to
`python ./PreprocessingKit/glassesCalibration/gazeMappingPipeline/tobii_preprocessing.py ../../../tobii_file/ ../../../data`

