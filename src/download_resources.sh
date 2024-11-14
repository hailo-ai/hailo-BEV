#!/usr/bin/bash
cd resources
mkdir models
cd models
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/normalized/petrv2_repvggB0_backbone_pp_800x320.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_repvggB0_transformer_pp_800x320.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/models/petrv2_postprocess.onnx
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/demos/BEV-demo/matmul.npy
cd ../

mkdir input
cd input
mkdir map_files
cd ../../
