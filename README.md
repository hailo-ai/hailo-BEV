# BEV Demo


This demo uses a Hailo-8 device with PETRv2 to process 6 input images from nuScenes dataset.
It annotates these images with 3D bounding boxes and creates Bird's Eye View (BEV) representations.

![Example](./resources/bev.gif)
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">
    <span property="dct:title">This gif uses images from <a href="https://www.nuscenes.org" target="_blank" rel="noopener noreferrer">nuScenes dataset</a>, and it</span> is licensed under
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0
        <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="">
        <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="">
        <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt="">
        <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt="">
    </a>
</p>


Pipeline
--------

![Pipeline](./resources/pipeline.png)

To more closely emulate a real application, run the demo with the 6 .jpg images from nuScenes.

When running the demo with raw data (.npy files) input, memory usage will be higher, as .npy files are loaded into RAM.
However, CPU usage will be lower because data loading does not occur in real time.


Requirements
------------

- hailo_platform==4.19.0
- Pyhailort


# Install the Demo - X86
-------------------------

1. Clone the repository:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-BEV.git
            
    cd BEV_Demo
    ```

2. Install dependencies:

    We recommend running it within a virtual environment.
    ```shell script
    pip install -r requirements.txt
    ```

3. Download demo resources:
    ```shell script
    ./download_resources.sh
    ```

4. Data creation:
    
    Download the mini dataset from https://www.nuscenes.org/nuscenes#download
    
    To run the demo with **.jpg** input, use:

    ```shell script
    ./src/common/prepare_data.py --data <path to nuScenes dataset>
    ```
    
    To run the demo with **raw data** input, use:

    ```shell script
    ./src/common/prepare_data.py --data <path to nuScenes dataset> --raw-data
    ```

# Install the Demo - Embedded Architecture
------------------------------------------

1. Clone the repository both on the **host** and on the **platform**:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-BEV.git
            
    cd BEV_Demo
    ```

2. Install dependencies both on the **host** and on the **platform**:

    We recommend running it within a virtual environment.
    ```shell script
    pip install -r requirements.txt
    ```

3. Download demo resources on the platform only:
    ```shell script
    ./download_resources.sh
    ```

4. Data creation:
    
    Download the mini dataset from https://www.nuscenes.org/nuscenes#download
    
    **Execute this command on the **host** only:**
    
    To run the demo with **.jpg** input, use:

    ```shell script
    ./src/common/prepare_data.py --data <path to nuScenes dataset>
    ```
    
    To run the demo with **raw data** input, use:

    ```shell script
    ./src/common/prepare_data.py --data <path to nuScenes dataset> --raw-data
    ```
    After running the command on the **host**, copy the /resources/input/ directory to the platform.
    - **Note**: The /resources/input/ folder must be present on both the host and the platform.

# Run Inference - X86
---------------------
To run inference, execute the following command from the repository's main folder:
```shell script
./src/x86/bev.py -d <data_path> --run-slow(optional)
```
For optimal visibility of 3D boxes, use the --run-slow flag. This will run the demo at 5 FPS. Since nuScenes samples were captured at 2 FPS, higher FPS values might obscure the 3D box display or create an illusion of faster vehicle movement.

**Arguments**

```shell script
./src/x86/bev.py -h
```

- ``-i, --input``: Path to the input folder, Use this flag only if you have modified the default input folder location.
- ``-m, --models``: Path to the models folder, Use this flag only if you have modified the default model folder location.
- ``-d, --data``: Path to the data folder, where the nuScenes dataset is.
- ``--run-slow``: Run the demo at 5 FPS for better visualization of 3D boxes.
- ``--raw-data``: Run the demo from raw data for lower cpu usage.


**Example**

```shell script
./src/x86/bev.py --run-slow
```
*Press Ctrl-C to stop the demo.*
# Run Inference - Embedded Architecture
-----------------------------------------

Platform side
-------------

To run inference, execute the following command from the repository's main folder:
```shell script
./src/embedded/platform/bev.py -d <data_path> --set-ip <platform ip> --set-port <communication port>(optional) --run-slow(optional)
```
For optimal visibility of 3D boxes, use the --run-slow flag. This will run the demo at 5 FPS. Since nuScenes samples were captured at 2 FPS, higher FPS values might obscure the 3D box display or create an illusion of faster vehicle movement.


**Arguments**

```shell script
./src/embedded/platform/bev.py -h
```

- ``-i, --input``: Path to the input folder, Use this flag only if you have modified the default input folder location.
- ``-m, --models``: Path to the models folder, Use this flag only if you have modified the default model folder location.
- ``-d, --data``: Path to the data folder, where the nuScenes dataset is.
- ``--run-slow``: Makes the demo run in 5 FPS.
- ``--jpg-input``: Run the demo using .jpg data input, may require more CPU power.
- ``--set-ip``: Set platform's ip.
- ``--set-port``: Change the port from 5555 to another one.

**Example**

```shell script
./src/embedded/platform/bev.py --set-ip <platform ip> --run-slow 
```
*Press Ctrl-C on the host side only to stop the demo.*

Host side 
----------
Run the Host side only after you see "Listening on {ip}:{port}" on the platform side.

Run visualization: 
```shell script
./src/embedded/host/viz.py -d <data_path> --set-ip <platform ip> --set-port <communication port>(optional)
```

**Arguments**
```shell script
./src/embedded/host/bev.py -h
```

- ``-d, --data``: Path to the data folder, where the nuScenes dataset is.
- ``-i, --input``: Path to the input folder, Use this flag only if you have modified the default input folder location.
- ``--set-ip``: Set platform's ip.
- ``--set-port``: Change the port from 5555 to another one.


**Example**

```shell script
./src/embedded/host/bev.py --set-ip <platform ip>
```
*Press Ctrl-C to stop the demo.*

Additional Notes
----------------
- Ran the demo on: Dell PC (Model: Latitude 5431), with CPU (Model: 12th Gen Intel(R) Core(TM) i7-1270P).
- All data is computed at runtime, except of the map, which is derived from the ground truth.

License
----------
The BEV Demo is released under the MIT license. Please see the https://github.com/hailo-ai/hailo-BEV/blob/main/LICENSE file for more information.


Disclaimer
----------
This code demo is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code demo. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code demo or any part of it. If an error occurs when running this demo, please open a ticket in the "Issues" tab.

This demo was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The demo might work for other versions, other environment or other HEF file, but there is no guarantee that it will.