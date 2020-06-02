# jhu-eeg
JHU EEG

Installation:
-----
Clone the repository using ```$ git clone https://github.com/jcraley/jhu-eeg.git```  

Python >= 3.5 is required. Other packages can be installed by creating a virtual environment and using the provided requirements.txt file.

To create the virtual environment:  
```
$ python3 -m venv eeg-gui-venv
``` 

Activate the environment:  
```
$ source eeg-gui-venv/bin/activate
```  

Install required packages:  
```
$ pip install numpy==1.18.1  
$ pip install -r requirements.txt
```


Running the visualizer:
-----
You can then run the visualizer from the main folder using  
    ```python visualization/plot.py```
    
For more command line options, see the section below. 

If you get the error "ModuleNotFoundError: No module named 'preprocessing' "
this is likely a path issue with python, and can be fixed using
    ```export PYTHONPATH=$(pwd)```
    
Features:
-----
***EDF files:***  
Average reference and longitudinal bipolar montages with the typical channel naming conventions are supported. Other channels can be plotted but will not be considered part of the montage. 

***Loading predictions:***  
Predictions can be loaded as pytorch (.pt) files or using preprocessed data and a model (also saved as .pt files). In both cases, the output is expected to be of length (k * number of samples in the edf file). Channel-wise predictions will be plotted starting from the top of the screen. 

***Saving to .edf:***  
This will save the signals that are currently being plotted. If the signals are filtered and predictions are plotted, filtered signals will be saved and predictions will be saved as well. 

***Saving to .png:***  
This will save an image of the current graph along with any predictions that are plotted. 

![](visualizerImg.png)

Command line options:
-----
We have added command line options to streamline use: 
```
$ python visualization/plot.py --show {0 | 1} --fn [EDF_FILE] --montage-file [TXT_FILE] 
--predictions-file [PT_FILE] --export-png-file [PNG_FILE] --filter [LOW_PASS_FS] [HIGH_PASS_FS] [NOTCH_FS] 
--location [INT] --window-width {5 | 10 | 15 | 20 | 25 | 30}
```
These options include:

* Whether or not to show the visualizer
* The .edf file to load
* What montage to use
* Predictions
* Name of .png file to save the graph
* Filter specifications
* Where in time to load the graph
* How many seconds to show in the window
