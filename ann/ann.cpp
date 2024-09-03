#include "../layer/inputLayer.cpp"
#include "../layer/hiddenLayer.cpp"
#include "../layer/outputLayer.cpp"
#include "../utils/json.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <fstream>
using json = nlohmann::json;
using namespace std;
enum AnnMode{initializing,training,testing,production};
struct LayerInfo{
  int neuronCount=1;
  ActivationTypes activationType=unipolarSigmoidal;
};
class Ann {
public:
    string inputFile;
    string trainingFile;
    string testingFile; 
    string outputFile;
    int inputVectorDimension;
    int outputVectorDimension;
    int numberOfPresentations;
    int hiddenLayerCount;
    int presentationNo;
    int currentIteration;
    vector<LayerInfo>layerInfo;
    vector<Layer*>layers;
    vector<vector<float> > inputs;
    vector<vector<float> > outputs;
    vector<pair<float,float> > inputMaximaMinima;
    vector<pair<float,float> >outputMaximaMinima;
    float thisIterationError;
    AnnMode mode;
    json trainedWeightsAndBias;
    float permissableError;
    string err;
    float trainRatio;
    Ann(string inputFile, int inputVectorDimension, int outputVectorDimension,AnnMode mode,vector<LayerInfo> layerInfo={},float permissableError=0.001,float trainRatio=0.8);
    Ann(){};
    void buildAnn();
    void calcOutput();
    void printVectors();
    void printLayers();
    Vector getPreviousLayerOutput(Layer* lyr);
    void calcDelEAndW();
    void updateNeurons();
    void train();
    void test();
    void segregateData(float segregationRatio=0.8F);
    float calcError();
    void writeErrorsInFile();
    void saveAnn();
    void readDataFile();
    void readAnn();
    void normalizeData();
    void normalizeOutput();
    void normalizeInputs();
    bool normalizeSingleInput(vector<float> singleInput);
    void printOutput();
    void start();
    vector<float> unNormalizedOutput(vector<float> normalizedOutputs);
};
Ann::Ann(string inputFile,  int inputVectorDimension, int outputVectorDimension, AnnMode mode,vector<LayerInfo> layerInfo,float permissableError,float trainRatio)
    :permissableError(permissableError), inputFile(inputFile),mode(mode), inputVectorDimension(inputVectorDimension), outputVectorDimension(outputVectorDimension),layerInfo(layerInfo),trainRatio(trainRatio){

    if (inputVectorDimension <= 0 || outputVectorDimension <= 0) {
        throw invalid_argument("Dimensions must be positive integers.");
    }
    outputFile="trainedModel/ann.json";
    trainingFile="normalizedData/training.json";
    testingFile="normalizedData/testing.json";
    presentationNo=0;
    thisIterationError=1e20;
    if(mode==training){
      readDataFile();
    inputs.resize(numberOfPresentations, vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(numberOfPresentations, vector<float>(outputVectorDimension, 0.0f));
    }
    else if(mode==testing){
      layerInfo={};
      readDataFile();
    inputs.resize(numberOfPresentations, vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(numberOfPresentations, vector<float>(outputVectorDimension, 0.0f));
      readAnn();
      json layerInfoJson=trainedWeightsAndBias["layerInfo"];
      for(size_t i=0;i<layerInfoJson.size();i++){
        LayerInfo lyr;
        lyr.activationType=layerInfoJson[i]["activationType"];
        lyr.neuronCount=layerInfoJson[i]["neuronCount"];
        layerInfo.push_back(lyr);
      }
    }
    else if(mode==production){
      readAnn();
    }
    hiddenLayerCount=layerInfo.size();
    buildAnn();
}
void Ann::buildAnn(){
  if(mode==training){
    json config;
      std::ifstream inFile("normalizedData/config.json");
        if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file ann.json");
    }
    inFile >>config;
     inputMaximaMinima.resize(inputVectorDimension);
        outputMaximaMinima.resize(outputVectorDimension);

        for (size_t i = 0; i < inputVectorDimension; ++i) {
            inputMaximaMinima[i].first = config["inputMaximaMinima"][i]["first"];
            inputMaximaMinima[i].second = config["inputMaximaMinima"][i]["second"];
        }

        for (size_t i = 0; i < outputVectorDimension; ++i) {
            outputMaximaMinima[i].first = config["outputMaximaMinima"][i]["first"];
            outputMaximaMinima[i].second = config["outputMaximaMinima"][i]["second"];
        }

  layers.push_back(new InputLayer(inputVectorDimension,0));
  layers[0]->buildLayer(1,nothing);
  for(int i=1;i<=hiddenLayerCount;i++){
    layers.push_back(new HiddenLayer(layerInfo[i-1].neuronCount,i));
    layers[i]->buildLayer(layers[i-1]->neuronCount,layerInfo[i-1].activationType);
  }
  layers.push_back(new OutputLayer(outputVectorDimension,hiddenLayerCount+1));
  layers[hiddenLayerCount+1]->buildLayer(layers[hiddenLayerCount]->neuronCount);
  }
  else if(mode==testing||mode==production){
    layers.push_back(new InputLayer(inputVectorDimension,0));
    layers[0]->buildLayer(trainedWeightsAndBias["layers"][0],nothing);
     for(size_t i=1;i<=hiddenLayerCount;i++){
    layers.push_back(new HiddenLayer(layerInfo[i-1].neuronCount,i));
    layers[i]->buildLayer(trainedWeightsAndBias["layers"][i],layerInfo[i-1].activationType);
     }
  layers.push_back(new OutputLayer(layerInfo[hiddenLayerCount-1].neuronCount,hiddenLayerCount+1));
  layers[hiddenLayerCount+1]->buildLayer(trainedWeightsAndBias["layers"][hiddenLayerCount+1]);
   inputMaximaMinima.resize(inputVectorDimension);
        outputMaximaMinima.resize(outputVectorDimension);

        for (size_t i = 0; i < inputVectorDimension; ++i) {
            inputMaximaMinima[i].first = trainedWeightsAndBias["inputMaximaMinima"][i]["first"];
            inputMaximaMinima[i].second = trainedWeightsAndBias["inputMaximaMinima"][i]["second"];
        }

        for (size_t i = 0; i < outputVectorDimension; ++i) {
            outputMaximaMinima[i].first = trainedWeightsAndBias["outputMaximaMinima"][i]["first"];
            outputMaximaMinima[i].second = trainedWeightsAndBias["outputMaximaMinima"][i]["second"];
        }
  }
};

void Ann::readAnn() {
      std::ifstream inFile("trainedModel/ann.json");
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file ann.json");
    }
    inFile >> trainedWeightsAndBias;
    inFile.close();
}

void Ann::printLayers(){
  for(int i=0;i<hiddenLayerCount+2;i++){
    for(int j=0;j<layers[i]->neuronCount;j++){
      layers[i]->layerNeurons[j].print();
    }
  }
  
}
void Ann::readDataFile() {
  string fileToread=mode==training?trainingFile:testingFile;
    ifstream inputFileStream(fileToread);
    if (!inputFileStream.is_open()) {
        throw runtime_error("Could not open the file: " + inputFile);
    }
    json jsonData;
    inputFileStream >> jsonData;
    // cout<<jsonData.size();

  if (!jsonData.is_array()) {
        throw runtime_error("Expected an array of objects in the JSON file.");
    }
    int totalData=jsonData.size();
    inputs.resize(totalData, std::vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(totalData, std::vector<float>(outputVectorDimension, 0.0f));
    for (size_t i = 0; i < jsonData.size(); ++i) {
        if (!jsonData[i].contains("inputs") || !jsonData[i].contains("outputs")) {
            throw runtime_error("Each JSON object must contain 'inputs' and 'outputs' arrays.");
        }
      vector<float> inputVector = jsonData[i]["inputs"].get<vector<float> >();
        vector<float> outputVector = jsonData[i]["outputs"].get<vector<float> >();
      if (inputVector.size() != inputVectorDimension) {
            throw runtime_error("Input vector size does not match expected input dimension.");
        }
        if (outputVector.size() != outputVectorDimension) {
            throw runtime_error("Output vector size does not match expected output dimension.");
        }
        inputs[i] = inputVector;
        outputs[i] = outputVector;
    }
    numberOfPresentations=jsonData.size();
}
void Ann::calcOutput(){
  Vector input(inputVectorDimension);
  for(int i=0;i<inputVectorDimension;i++){
    vector<float>vec(1,inputs[presentationNo][i]);
    Vector v(vec);
    layers[0]->layerNeurons[i].calcNeti(v);
    input[i]=layers[0]->layerNeurons[i].calcFNeti();
  }
  for(int i=1;i<hiddenLayerCount+2;i++){
  Vector nextInput(layers[i]->neuronCount);
    for(int j=0;j<layers[i]->neuronCount;j++){
      layers[i]->layerNeurons[j].calcNeti(input);
      nextInput[j]= layers[i]->layerNeurons[j].calcFNeti();
    }
    input=nextInput;
  }
}
void Ann::printVectors() {
    cout << "Inputs:" << endl;
    for (const auto& inputVector : inputs) {
        for (float value : inputVector) {
            cout << value << " ";
        }
        cout << endl;
    }
    cout << "Outputs:" << endl;
    for (const auto& outputVector : outputs) {
        for (float value : outputVector) {
            cout << value << " ";
        }
        cout << endl;
    }
}
void Ann::saveAnn() {
    json annToSave;
    annToSave["inputVectorDimension"] = inputVectorDimension;
    annToSave["outputVectorDimension"] = outputVectorDimension;
    annToSave["numberOfPresentations"] = numberOfPresentations;
    annToSave["hiddenLayerCount"] = hiddenLayerCount;
    annToSave["presentationNo"] = presentationNo;
    annToSave["currentIteration"] = currentIteration;
    annToSave["thisIterationError"] = thisIterationError;
    annToSave["layerInfo"] = json::array();
    for (const auto& info : layerInfo) {
        json layerInfoJson;
        layerInfoJson["neuronCount"] = info.neuronCount;
        layerInfoJson["activationType"] = info.activationType;  
        annToSave["layerInfo"].push_back(layerInfoJson);
    }
        annToSave["inputMaximaMinima"] = json::array();
    for (const auto& pair : inputMaximaMinima) {
      
        json pairJson;
        pairJson["first"] = pair.first;
        pairJson["second"] = pair.second;
        annToSave["inputMaximaMinima"].push_back(pairJson);
    }
    annToSave["outputMaximaMinima"] = json::array();
    for (const auto& pair : outputMaximaMinima) {
        json pairJson;
        pairJson["first"] = pair.first;
        pairJson["second"] = pair.second;
        annToSave["outputMaximaMinima"].push_back(pairJson);
    }
    annToSave["layers"] = json::array();
    for (const auto& layer : layers) {
      json layerJson=layer->parseLayer();
        annToSave["layers"].push_back(layerJson);
    }
    std::ofstream outFile(outputFile);
    if (outFile.is_open()) {
        outFile << annToSave.dump(4);  
        outFile.close();
    } else {
        std::cerr << "Error opening file for writing JSON!" << std::endl;
    }
}

Vector Ann::getPreviousLayerOutput(Layer * lyr){
   Layer* previousLayer=layers[lyr->layerNumber-1];
   Vector output(previousLayer->neuronCount);
   for(int i=0;i<previousLayer->neuronCount;i++){
     output[i]=previousLayer->layerNeurons[i].fNeti;
   }
return output;
}
void Ann::calcDelEAndW(){
  Vector desiredOutput=Vector(outputs[presentationNo]);
  layers[hiddenLayerCount+1]->computeDelE(desiredOutput);
  Vector previousOutput=getPreviousLayerOutput(layers[hiddenLayerCount+1]);
  layers[hiddenLayerCount+1]->computeDelW(previousOutput);
  Vector passBackError=layers[hiddenLayerCount+1]->passBackError();
  for(int i=hiddenLayerCount;i>0;i--){
    layers[i]->computeDelE(passBackError);
    previousOutput=getPreviousLayerOutput(layers[i]);
    layers[i]->computeDelW(previousOutput);
    passBackError=layers[i]->passBackError();
  }
}
void Ann::updateNeurons(){
  for(int i=1;i<hiddenLayerCount+2;i++){
    for(int j=0;j<layers[i]->neuronCount;j++){
      layers[i]->layerNeurons[j].update();
    }
  }
}
float Ann::calcError(){
float thisOutputErr=0;
for(int i=0;i<outputVectorDimension;i++){
  float outp=layers[hiddenLayerCount+1]->layerNeurons[i].fNeti;
  thisOutputErr+= pow(outp-outputs[presentationNo][i],2);
}
return thisOutputErr;}
void Ann::train(){
  while(thisIterationError>permissableError){
    presentationNo=0;
    thisIterationError=0;
    currentIteration++;
    for(int j=0;j<numberOfPresentations;j++){
    calcOutput();
    calcDelEAndW();
    updateNeurons();
   thisIterationError+=calcError();
   ++presentationNo;
    }
    thisIterationError=(thisIterationError/numberOfPresentations)*100;
    cout<<"Current Error = "<<thisIterationError<<"%\n";
  }
  saveAnn();
}
void Ann::test(){
  float iterationError=0;
  for(int j=0;j<numberOfPresentations;j++){
    calcOutput();
    iterationError+=calcError();
   ++presentationNo;
  }
    cout<<"Error in testing Data = "<<(iterationError/numberOfPresentations)*100<<"%\n";
  
}
void Ann::writeErrorsInFile(){
    std::ofstream outFile;
    std::string filename = "output.txt";
    outFile.open(filename);
    if (!outFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return ;
    }
    outFile << err<< std::endl;
    outFile.close();

}
void Ann::normalizeOutput(){
    int m = numberOfPresentations;
  int  n = outputVectorDimension;
    outputMaximaMinima.resize(n);
    for (int i = 0; i < n; ++i) {
        outputMaximaMinima[i] = {outputs[0][i], outputs[0][i]};
        for (int j = 1; j < m; ++j) {
            outputMaximaMinima[i].first = min(outputMaximaMinima[i].first, outputs[j][i]);
            outputMaximaMinima[i].second = max(outputMaximaMinima[i].second, outputs[j][i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            outputs[j][i] = (outputs[j][i] - outputMaximaMinima[i].first) / 
                            (outputMaximaMinima[i].second - outputMaximaMinima[i].first);
        }
    }

    
}
void Ann::normalizeInputs() {
    int m = numberOfPresentations;
    int n = inputVectorDimension;
    inputMaximaMinima.resize(n);
    for (int i = 0; i < n; ++i) {
        inputMaximaMinima[i] = {inputs[0][i], inputs[0][i]};
        for (int j = 1; j < m; ++j) {
            inputMaximaMinima[i].first = min(inputMaximaMinima[i].first, inputs[j][i]);
            inputMaximaMinima[i].second = max(inputMaximaMinima[i].second, inputs[j][i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            inputs[j][i] = (inputs[j][i] - inputMaximaMinima[i].first) / 
                           (inputMaximaMinima[i].second - inputMaximaMinima[i].first) ;
        }
    }

}
void Ann::normalizeData(){
  normalizeInputs();
  normalizeOutput();
}
bool Ann::normalizeSingleInput(vector<float> input){
inputs=vector<vector<float> >(1,vector<float>(inputVectorDimension));
for (int i = 0; i < inputVectorDimension; ++i) {
  if(input[i]<inputMaximaMinima[i].first||input[i]>inputMaximaMinima[i].second){
    cout<<
"Sorry, I can't extrapolate. Please provide input components within the trained range.\n";
return false;
  }
            inputs[0][i] = (input[i] - inputMaximaMinima[i].first) / 
                           (inputMaximaMinima[i].second - inputMaximaMinima[i].first) ;
    }
    return true;
}
vector<float> Ann::unNormalizedOutput(vector<float> normalizedOutput){
  vector<float> unNormalized;
  for(int i=0;i<outputVectorDimension;i++){
  float min=  outputMaximaMinima[i].first;
  float max=  outputMaximaMinima[i].second;
  unNormalized.push_back(normalizedOutput[i]*(max-min)+min);
  }
  return unNormalized;
}
void Ann::segregateData(float trainRatio) {
    std::ifstream inputFileStream(inputFile);
    if (!inputFileStream.is_open()) {
        throw std::runtime_error("Could not open the input file: " + inputFile);
    }

    json jsonData;
    inputFileStream >> jsonData;
    inputFileStream.close();

    if (!jsonData.is_array()) {
        throw std::runtime_error("Input file should contain a JSON array.");
    }
    std::vector<std::pair<std::vector<float>, std::vector<float>>> allData;

    for (size_t i = 0; i < jsonData.size(); ++i) {
        if (!jsonData[i].contains("inputs") || !jsonData[i].contains("outputs")) {
            throw std::runtime_error("Each JSON object must contain 'inputs' and 'outputs' arrays.");
        }

        std::vector<float> inputVector = jsonData[i]["inputs"].get<std::vector<float>>();
        std::vector<float> outputVector = jsonData[i]["outputs"].get<std::vector<float>>();

        if (inputVector.size() != static_cast<size_t>(inputVectorDimension)) {
            throw std::runtime_error("Input vector size does not match expected input dimension.");
        }

        if (outputVector.size() != static_cast<size_t>(outputVectorDimension)) {
            throw std::runtime_error("Output vector size does not match expected output dimension.");
        }

        allData.emplace_back(inputVector, outputVector);
    }

    size_t totalData = allData.size();
    if (totalData == 0) {
        throw std::runtime_error("No data found in the input file.");
    }
    inputs.resize(totalData, std::vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(totalData, std::vector<float>(outputVectorDimension, 0.0f));

    for (size_t i = 0; i < totalData; ++i) {
        inputs[i] = allData[i].first;
        outputs[i] = allData[i].second;
    }
    numberOfPresentations=totalData;
    normalizeData();

    std::vector<size_t> indices(totalData);
    for (size_t i = 0; i < totalData; ++i) {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    size_t trainSize = static_cast<size_t>(trainRatio * totalData);
    size_t testSize = totalData - trainSize;

    json trainingJson = json::array();
    json testingJson = json::array();

    for (size_t i = 0; i < trainSize; ++i) {
        size_t idx = indices[i];
        json dataPoint;
        dataPoint["inputs"] = inputs[idx];
        dataPoint["outputs"] = outputs[idx];
        trainingJson.push_back(dataPoint);
    }

    for (size_t i = trainSize; i < totalData; ++i) {
        size_t idx = indices[i];
        json dataPoint;
        dataPoint["inputs"] = inputs[idx];
        dataPoint["outputs"] = outputs[idx];
        testingJson.push_back(dataPoint);
    }
    std::ofstream trainingOut(trainingFile);
    if (!trainingOut.is_open()) {
        throw std::runtime_error("Could not open training file for writing: " + trainingFile);
    }
    trainingOut << trainingJson.dump(4);
    trainingOut.close();

    std::ofstream testingOut(testingFile);
    if (!testingOut.is_open()) {
        throw std::runtime_error("Could not open testing file for writing: " + testingFile);
    }
    testingOut << testingJson.dump(4);
    testingOut.close();

    std::cout << "Data segregation completed successfully.\n";
    std::cout << "Training data size: " << trainSize << "\n";
    std::cout << "Testing data size: " << testSize << "\n";
    json config;
    for(int i=0;i<inputVectorDimension;i++){
      json obj;
      obj["first"]=inputMaximaMinima[i].first;
      obj["second"]=inputMaximaMinima[i].second;
      config["inputMaximaMinima"].push_back(obj);
    }
    for(int i=0;i<outputVectorDimension;i++){
      json obj;
      obj["first"]=outputMaximaMinima[i].first;
      obj["second"]=outputMaximaMinima[i].second;
      config["outputMaximaMinima"].push_back(obj);
    }
    std::ofstream configOut("normalizedData/config.json");
    configOut << config.dump(4);
    configOut.close();

}
void Ann:: printOutput(){
  vector<float> outs;
  for(int i=0;i<outputVectorDimension;i++){
    outs.push_back(layers[hiddenLayerCount+1]->layerNeurons[i].fNeti);
  }
  vector<float> f=unNormalizedOutput(outs);
  cout<<"[";
  for(int i=0;i<outputVectorDimension;i++){
    cout<<f[i]<<", ";
  }
  cout<<"]\n";
}
void Ann::start(){
  if(mode==initializing)
    segregateData(trainRatio);
  else if(mode==training)
   train();
   else if(mode==testing)
   test();
   else if(mode==production)
   {
    int testCases;
    cout<<"Enter the number of test cases you want to calculate output: ";
    cin>>testCases;
      while(testCases--){
  vector<float> input(inputVectorDimension);
  cout<<"Enter the inputs seperated by spaces: ";
    for(int i = 0; i < inputVectorDimension; i++){
        cin>>input[i];
    }
  bool isNormalized=  normalizeSingleInput(input);
    if(isNormalized)
    {calcOutput();
printOutput();}
}
   }
}
