# D415-OPENVINO
This repo demos real-time face mask detection / human pose estimation with an Intel D415

## SETUP
- clone the submodule necessary for face mask detection:
  -  ```
     git submodule init
     git submodule update
     ``` 
- create python virtual environment
- install dependencies: `pip install -r requirements.txt`

## USAGE

### PYTHON DEMOS
- ensure openvino environment variables are set and python environment is activated
  ```
  . /opt/intel/openvino/bin/setupvars.sh
  . .venv/bin/activate
  ```
- to run openvino mask detection:
   - `python mask_detection.py --device CPU`
- to run openvino human pose estimation demo:
   - FP16: `python human_pose.py /opt/intel/openvino/deployment_tools/open_model_zoo/models/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml /opt/intel/openvino/deployment_tools/open_model_zoo/models/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin`
   - FP16-int8: `python human_pose.py /opt/intel/openvino/deployment_tools/open_model_zoo/models/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml /opt/intel/openvino/deployment_tools/open_model_zoo/models/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.bin`
