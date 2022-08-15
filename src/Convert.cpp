# include "Convert.h"
# include "utility.h"

bool Convert::buildEngine(const string& onnxModelPath, const string & save_engine_path){
    /*/ input: onnxModelPath  /*/
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if(!builder){
        return false;
    }
    // the settting is for onnx
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){
        return false;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if(!parser){
        return false;
    }
    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return false;
    }
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        return false;
    }

    cout<<2*(1ULL<<32)<<endl;
    config->setMaxWorkspaceSize(2*(1ULL<<32));

    if (precision_== "fp16"){
        config->setFlag(BuilderFlag::kFP16);
    }
    builder->setMaxBatchSize(maxBatchSize);

    const auto input = network->getInput(0);
    const auto inputName = input->getName();

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4{MIN_DIMS_[0], MIN_DIMS_[1], MIN_DIMS_[2], MIN_DIMS_[3]});
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4{OPT_DIMS_[0], OPT_DIMS_[1], OPT_DIMS_[2], OPT_DIMS_[3]});
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4{MAX_DIMS_[0], MAX_DIMS_[1], MAX_DIMS_[2], MAX_DIMS_[3]});
    config->addOptimizationProfile(profile);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    std::ofstream fout(save_engine_path.c_str(), std::ofstream::binary);
    if (!fout) {
        std::cerr << "could not open engine output file, saveEngine failed!" << std::endl;
        return false;
    }
    fout.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Success, save engine to " << save_engine_path << std::endl;
    return true;
}


bool Convert::deserializeEngine(const string& engineName){
    std::ifstream file(engineName, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineName << " error!" << std::endl;
        return false;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr);
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    assert(context != nullptr);

    delete[] trtModelStream;
    return true;
}

int Convert::Model_Init(const string& engine_path, const string &onnx_path, const string &save_engine_dir){
    string our_engine_path = "";
    if (OCR::Utility::PathExists(engine_path)){
        our_engine_path = engine_path;
        cout << "[" << engine_path << "] exists, start to read engine file: " << endl;
    }else{
        cout << "[" << engine_path << "] does not exist, we will use [" << onnx_path<< "] to build engine." << endl;
        if (OCR::Utility::PathExists(engine_path)){
            cout<<" onnx file does not exist, please check your onnx path: [" << onnx_path<< "]"<<endl;
        }else{
            if (!OCR::Utility::PathExists(save_engine_dir)) {
                OCR::Utility::CreateDir(save_engine_dir);
            }
            string onnx_base_name = OCR::Utility::basename(onnx_path);
            auto index = onnx_base_name.find_last_of(".");
            our_engine_path = save_engine_dir + "/" + onnx_base_name.substr(0, index) + "_" + precision_ + ".engine";
            if(buildEngine(onnx_path, our_engine_path))
                cout<<" Sucessfully build [" << our_engine_path << "] engine from onnx." <<endl;
            else{
                cout<<" error in build engine from onnx" <<endl;
                return -1;
            }
        }
    }
    if(deserializeEngine(our_engine_path))
        cout<< " Sucessfully deserialize [" << our_engine_path<< "]" << endl;
    else{
        cout<<" deserialize [" << our_engine_path<< "] failed, model init failed"<<endl;
        return -2;
    }
    assert(engine->getNbBindings() == 2); //check if is a input and a output
    return 0;
}

Convert::~Convert(){
}