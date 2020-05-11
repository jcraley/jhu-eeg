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

If you get the error "ModuleNotFoundError: No module named 'preprocessing' "
this is likely a path issue with python, and can be fixed using
    ```export PYTHONPATH=$(pwd)```
    
Features:
-----
***EDF files:***  
Average reference and longitudinal bipolar montages with the typical channel naming conventions are supported. Other channels can be plotted but will not be part of the montage. 

***Loading predictions:***  
Predictions can be loaded as pytorch (.pt) files or using preprocessed data and a model (also saved as .pt files). In both cases, the output is expected to be of length (k * number of samples in the edf file). The second dimension can be either 1 or 2 if predictions are for all channels, or the number of channels for channel-wise predictions. For channel-wise predictions, it will be assumed that the channels are in the same order as are plotted in the visualizer. 

***Saving to .edf:***  
This will save the signals that are currently being plotted. If the signals are filtered and predictions are plotted, filtered signals will be saved and predictions will be saved as well. 
