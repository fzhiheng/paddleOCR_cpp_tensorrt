// create by zwy 2021,12,10
#pragma once

# include "NvInfer.h"
# include "logging.h"
# include "cuda_runtime_api.h"
# include "NvOnnxParser.h"

# include <iostream>
# include <fstream>
# include <unistd.h>
# include <string>

using namespace nvinfer1;
using namespace std;


// Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1<<31; // 30 =1 GB 1ULL << 34
const int maxBatchSize = 1;


class Convert{
public:
    Convert(){};
    Convert(const string & precision){this->precision_ = precision;};
    Convert(vector<int> MIN_DIMS, vector<int> OPT_DIMS, vector<int> MAX_DIMS){
        this->MIN_DIMS_ = MIN_DIMS;
        this->OPT_DIMS_ = OPT_DIMS;
        this->MAX_DIMS_ = MAX_DIMS;
    };
    Convert(vector<int> MIN_DIMS, vector<int> OPT_DIMS, vector<int> MAX_DIMS, const string & precision){
        this->precision_ = precision;
        this->MIN_DIMS_ = MIN_DIMS;
        this->OPT_DIMS_ = OPT_DIMS;
        this->MAX_DIMS_ = MAX_DIMS;
    };

    // 通过onnx来创建engine,并将创建的engine保存
    bool buildEngine(const string& onnx_path, const string & engineName); /*/ input: onnxModelPath  /*/

    // 输入engine名，若不存在，buildEngine创建一个, 并写成文件。
    bool getEngine(const string& engine_path, const string & onnx_path);

    // 解engine
    bool deserializeEngine(const string& engine_path);

    int Model_Init(const string& engine_path, const string onnx_path=" ");

    int get_engine_max_batch(){return this->MAX_DIMS_[0];};

    ~Convert();

    Logger gLogger; //日志
    std::unique_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context = nullptr;

private:
    std::string precision_ = "fp32";
    vector<int> MIN_DIMS_ = {1, 3, 20, 20};
    vector<int> OPT_DIMS_ = {1, 3, 512, 512};
    vector<int> MAX_DIMS_ = {1, 3, 640, 640};
//    vector<int> MIN_DIMS_ = {1, 3, 416, 416};
//    vector<int> OPT_DIMS_ = {1, 3, 416, 416};
//    vector<int> MAX_DIMS_ = {1, 3, 416, 416};
};