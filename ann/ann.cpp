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
enum AnnMode{training,testing,production};
struct LayerInfo{
  int neuronCount=1;
  ActivationTypes activationType=unipolarSigmoidal;
};
class Ann {
public:
    string inputFile;
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
    Ann(string inputFile, int inputVectorDimension, int outputVectorDimension, int numberOfPresentations,AnnMode mode,vector<LayerInfo> layerInfo={}, string outputFile="trainedModel/ann.json",float permissableError=0.001);
    Ann(){};
    void buildAnn();
    void calcOutput();
    void printVectors();
    void computeErrors();
    void printLayers();
    Vector getPreviousLayerOutput(Layer* lyr);
    void calcDelEAndW();
    void updateNeurons();
    void train();
    void writeErrorsInFile();
    void saveAnn();
    void readInputFile();
    void readAnn();
    void normalizeData();
    void normalizeOutput();
    void normalizeInputs();
    void normalizeSingleInput(vector<float> singleInput);
    void printOutput();
    vector<float> unNormalizedOutput(vector<float> normalizedOutputs);
};
Ann::Ann(string inputFile,  int inputVectorDimension, int outputVectorDimension, int numberOfPresentations,AnnMode mode,vector<LayerInfo> layerInfo,string outputFile,float permissableError)
    :permissableError(permissableError), inputFile(inputFile),mode(mode), outputFile(outputFile), inputVectorDimension(inputVectorDimension), outputVectorDimension(outputVectorDimension), numberOfPresentations(numberOfPresentations),layerInfo(layerInfo){

    if (inputVectorDimension <= 0 || outputVectorDimension <= 0 || numberOfPresentations <= 0) {
        throw invalid_argument("Dimensions and number of presentations must be positive integers.");
    }
    inputs.resize(numberOfPresentations, vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(numberOfPresentations, vector<float>(outputVectorDimension, 0.0f));
    presentationNo=0;
    thisIterationError=1e20;
    if(mode==training){
      readInputFile();
    }
    else if(mode==testing){
      layerInfo={};
      readInputFile();
      readAnn();
      json layerInfoJson=trainedWeightsAndBias["layerInfo"];
      for(size_t i=0;i<layerInfoJson.size();i++){
        LayerInfo lyr;
        lyr.activationType=layerInfoJson[i]["activationType"];
        lyr.neuronCount=layerInfoJson[i]["neuronCount"];
        layerInfo.push_back(lyr);
      }
    }
    hiddenLayerCount=layerInfo.size();
    buildAnn();
    // cout<<hiddenLayerCount;
}
void Ann::computeErrors(){
   
}
void Ann::buildAnn(){
  if(mode==training){
  layers.push_back(new InputLayer(inputVectorDimension,0));
  layers[0]->buildLayer(1,nothing);
  for(int i=1;i<=hiddenLayerCount;i++){
    layers.push_back(new HiddenLayer(layerInfo[i-1].neuronCount,i));
    layers[i]->buildLayer(layers[i-1]->neuronCount,layerInfo[i-1].activationType);
  }
  layers.push_back(new OutputLayer(outputVectorDimension,hiddenLayerCount+1));
  layers[hiddenLayerCount+1]->buildLayer(layers[hiddenLayerCount]->neuronCount);
  }
  else if(mode==testing){
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
  cout<<"=================================================================================="<<endl;;
    for(int j=0;j<layers[i]->neuronCount;j++){
      layers[i]->layerNeurons[j].print();
    }
  }
  
}
void Ann::readInputFile() {
    ifstream inputFileStream(inputFile);
    if (!inputFileStream.is_open()) {
        throw runtime_error("Could not open the file: " + inputFile);
    }
    json jsonData;
    inputFileStream >> jsonData;
  if (!jsonData.is_array()) {
        throw runtime_error("Expected an array of objects in the JSON file.");
    }
    if (jsonData.size() != numberOfPresentations) {
        throw runtime_error("Number of presentations in JSON file does not match the expected value.");
    }

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
}
void Ann::calcOutput(){
  Vector input(inputVectorDimension);
  for(int i=0;i<inputVectorDimension;i++){
    // cout<<"input"<<inputs[presentationNo][i]<<endl;
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
void Ann::train(){
  while(thisIterationError>permissableError){
    presentationNo=0;
    thisIterationError=0;
    currentIteration++;
    for(int j=0;j<numberOfPresentations;j++){
    calcOutput();
    calcDelEAndW();
    updateNeurons();
  float outp=layers[hiddenLayerCount+1]->layerNeurons[0].fNeti;
   thisIterationError+=pow(outp-outputs[presentationNo][0],2);
   ++presentationNo;
    }
    cout<<thisIterationError<<"\n";
      err+="("+to_string(currentIteration)+","+to_string(thisIterationError)+"),";
  }
      writeErrorsInFile();
  saveAnn();
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
void Ann::normalizeSingleInput(vector<float> input){
inputs=vector<vector<float> >(1,vector<float>(inputVectorDimension));
for (int i = 0; i < inputVectorDimension; ++i) {
            inputs[0][i] = (input[i] - inputMaximaMinima[i].first) / 
                           (inputMaximaMinima[i].second - inputMaximaMinima[i].first) ;
    }
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
void Ann:: printOutput(){
  vector<float> outs;
  for(int i=0;i<outputVectorDimension;i++){
    outs.push_back(layers[hiddenLayerCount+1]->layerNeurons[i].fNeti);
  }
  vector<float> f=unNormalizedOutput(outs);
  cout<<"[";
  for(int i=0;i<outputVectorDimension;i++){
    cout<<f[i];
  }
  cout<<"]\n";
}
// int main(){
//   vector<LayerInfo> vec;
//   LayerInfo lyr;
//   for(int i=0;i<3;i++){
//     lyr.activationType=unipolarSigmoidal;
//     lyr.neuronCount=3;
//     vec.push_back(lyr);
//   }
//   int inputVectorDimensions=2;
//   Ann a("./inputs2.json",inputVectorDimensions,1,300,testing,vec);
//   // a.readFile();
//   // a.buildAnn();
//   // a.normalizeData();
//   // a.train();
//   // a.calcOutput();
//   // a.printLayers();
//     // a.printLayers();
//   while(true){
//   vector<float> input(inputVectorDimensions);
//     for(int i = 0; i < inputVectorDimensions; i++){
//         cin>>input[i];
//     }
//     a.normalizeSingleInput(input);
//     a.calcOutput();

// a.printOutput();
// // a.printLayers();
// }
//   // a.printLayers();
//   // a.printVectors();
//   // a.saveAnn();
//   // a.saveAnn();
//   // a.printVectors();
//   // Ann b=a;
//   // for(float i=0.15;i<60;i+=0.1){
//   // vector<vector<float>> inps={{i}};
//   // b.inputs=inps;
//   // b.presentationNo=0;
//   // b.calcOutput();
//   // b.printLayers();}
//   // Ann a;
//   // a.readAnn();
//   // a.printVectors();
// }