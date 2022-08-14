# paddOCR_cpp_tensorrt
## environment

```shell
cuda 10.2
cudnn 8.2.1
tentorrt 8.2.1
```

## build

```shell
mkdir build && cd build
cmake .. -DTensorRT_DIR=/usr/local/tensorrt -DOpenCV_DIR=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 
```

## run

### build engine from onnx and inference

```shell
./build/ocr --build_precision="fp16" 
--det_onnx_dir="../myModels/det.onnx" --det_engine_dir="../myEngines/det_fp16.engine"
--det_onnx_input_name="x" --det_onnx_output_name="save_infer_model/scale_0.tmp_1"
--rec_onnx_ dir="../myModels/ch_rec_v3.onnx"
--rec_engine_dir="../myEngines/ch_rec_fp16.engine"
--rec_onnx_input_name="x" --rec_onnx_output_name="softmax_5.tmp_0"
--rec_char_dict_path="../myModels/dict_txt/ppocr_keys_v1.txt"
--det=true --cls=false --rec=true  --image_dir="../testImgs/11.jpg"
```

### load engine and inference

```shell
./build/ocr 
--det_engine_dir="../myEngines/det_fp16.engine"
--det_onnx_input_name="x" --det_onnx_output_name="save_infer_model/scale_0.tmp_1"
--rec_engine_dir="../myEngines/ch_rec_fp16.engine"
--rec_onnx_input_name="x" --rec_onnx_output_name="softmax_5.tmp_0"
--rec_char_dict_path="../myModels/dict_txt/ppocr_keys_v1.txt"
--rec_batch_num=1
--det=true --cls=false --rec=true --image_dir="../testImgs11.jpg"
```
