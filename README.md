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

### build engine from onnx and inference (you only have onnx files)

```shell
./build/ocr --build_precision="fp16" 
--det_onnx_dir="../myModels/det.onnx" 
--rec_onnx_ dir="../myModels/ch_rec_v3.onnx"
--save_engine_dir="../myEngines/"
--rec_char_dict_path="../myModels/dict_txt/ppocr_keys_v1.txt"
--rec_batch_num=1
--det=true --cls=false --rec=true  
--image_dir="../testImgs/11.jpg" --output="./output/"
```

It will build a engine from your onnx file. In this example above, you will get a engine named `det_fp16.engine` in  `../myEngines/`. 
The detection and recognize results are in `./output/`. 
The rec_batch_num should be $\leq$ 8. If you want bigger batch size, please modify the MAX_DIMS_ in `./src/rec.h`.

### load engine and inference (you already have engines)

```shell
./build/ocr 
--det_engine_dir="../myEngines/det_fp16.engine"
--rec_engine_dir="../myEngines/ch_rec_v3_fp16.engine"
--rec_char_dict_path="../myModels/dict_txt/ppocr_keys_v1.txt"
--rec_batch_num=1
--det=true --cls=false --rec=true 
--image_dir="../testImgs/11.jpg" --output="./output/"
```
