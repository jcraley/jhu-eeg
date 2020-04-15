# jhu-eeg
JHU EEG

Installing required packages for the visualizer:
-----
There are several libraries that are needed, they should all be able to be installed using pip3. You will need python >= 3.5. Packages can be installed by creating a virtual environment and using the provided requirements.txt file.

To create the virtual environment:  
python3 -m venv eeg-gui-venv   
Activate the environment:  
source eeg-gui-venv/bin/activate  
Install required packages:  
pip install numpy==1.18.1  
pip install -r requirements.txt 


Running the visualizer:
-----
You can then run the visualizer from the main folder using  
    python visualization/plot.py

If you get the error "ModuleNotFoundError: No module named 'preprocessing'
this is likely a path issue with python, and can be fixed by doing
    export PYTHONPATH=$(pwd)
