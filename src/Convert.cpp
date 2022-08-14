# include "Convert.h"

bool Convert::buildEngine(const string& onnxModelPath, const string& engineName){
    /*/ input: onnxModelPath  /*/
//    std::string engineName_tmp = engineName;
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

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    const auto inputName = input->getName();

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4{MIN_DIMS_[0], MIN_DIMS_[1], MIN_DIMS_[2], MIN_DIMS_[3]});
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4{OPT_DIMS_[0], OPT_DIMS_[1], OPT_DIMS_[2], OPT_DIMS_[3]});
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4{MAX_DIMS_[0], MAX_DIMS_[1], MAX_DIMS_[2], MAX_DIMS_[3]});
    config->addOptimizationProfile(profile);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    std::ofstream fout(engineName.c_str(), std::ofstream::binary);
    if (!fout) {
        std::cerr << "could not open engine output file, saveEngine failed!" << std::endl;
        return false;
    }
    fout.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Success, saved engine to " << engineName << std::endl;
    return true;
}


bool Convert::getEngine(const string& engineName, const string& onnx_path){

    if(access(engineName.c_str(), F_OK ) == -1){  // 如果该engine文件不存在

        cout << engineName<< " engine file is not exist, need to be created" << endl;

        if(access(onnx_path.c_str(), F_OK ) == -1){ // 如果onnx文件不存在
            cout << onnx_path <<" onnx path is not exist, can't create a engine from it "<<endl;
            return false;
        }
        else{
            if(buildEngine(onnx_path, engineName))
                cout<<"sucessful from onnx create "<< engineName <<endl;
            else{
                cout<<"error in getting engine"<< engineName <<endl;
                return false;
            }
        }       
    }
    else
       cout<< engineName <<" already exist"<<endl; 
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


int Convert::Model_Init(const string& engine_path, string onnx_path){
    if(getEngine(engine_path, onnx_path)){
        if(deserializeEngine(engine_path))
            cout<< "Sucessful deserialize engine file!" << endl;
        else{
            cout<<"deserialize engine failed, model init failed"<<endl;
            return -2;
        }
    }
    else{
        cout<<"can't get engine file, model init failed"<<endl;
        return -1;
    }
    assert(engine->getNbBindings() == 2); //check if is a input and a output
    return 0;
}
Convert::~Convert(){
}