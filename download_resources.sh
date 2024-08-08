#!/usr/bin/bash
cd resources
mkdir models
cd models
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_b0_backbone_x32_BN_q_304_dec_3_UN_800x320.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_const0.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_middle_process_1.onnx
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_middle_process_2.onnx
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_post_process.onnx
cd ../

mkdir results
cd results
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/results/scenes_data.json
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/results/scene-0001.json
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/results/scene-0002.json
cd ../
