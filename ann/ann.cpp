#include "../layer/inputLayer.cpp"
#include "../layer/hiddenLayer.cpp"
#include "../layer/outputLayer.cpp"
#include "json.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <fstream>
using json = nlohmann::json;
using namespace std;
struct LayerInfo{
  int neuronCount=1;
  ActivationTypes activationType=unipolarSigmoidal;
};
class Ann {
private:
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
    float thisIterationError;
public:
    string err;
    Ann(string inputFile, string outputFile, int inputVectorDimension, int outputVectorDimension, int numberOfPresentations,vector<LayerInfo> layerInfo);
    Ann(){};
    void buildAnn();
    void readFile();
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
    void readAnn();
    void parseAnn();
};
Ann::Ann(string inputFile, string outputFile, int inputVectorDimension, int outputVectorDimension, int numberOfPresentations,vector<LayerInfo> layerInfo)
    : inputFile(inputFile), outputFile(outputFile), inputVectorDimension(inputVectorDimension), outputVectorDimension(outputVectorDimension), numberOfPresentations(numberOfPresentations),layerInfo(layerInfo){

    if (inputVectorDimension <= 0 || outputVectorDimension <= 0 || numberOfPresentations <= 0) {
        throw invalid_argument("Dimensions and number of presentations must be positive integers.");
    }
    inputs.resize(numberOfPresentations, vector<float>(inputVectorDimension, 0.0f));
    outputs.resize(numberOfPresentations, vector<float>(outputVectorDimension, 0.0f));
    hiddenLayerCount=layerInfo.size();
    presentationNo=0;
    thisIterationError=1e20;

}
void Ann::computeErrors(){
   
}
void Ann::buildAnn(){
  layers.push_back(new InputLayer(inputVectorDimension,0));
  layers[0]->buildLayer(1,nothing);
  for(int i=1;i<=hiddenLayerCount;i++){
    layers.push_back(new HiddenLayer(layerInfo[i-1].neuronCount,i));
    layers[i]->buildLayer(layers[i-1]->neuronCount,layerInfo[i-1].activationType);
  }
  layers.push_back(new OutputLayer(outputVectorDimension,hiddenLayerCount+1));
  layers[hiddenLayerCount+1]->buildLayer(layers[hiddenLayerCount]->neuronCount);
};
void Ann::saveAnn() {
    std::ofstream outFile("ann.dat", std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(&inputVectorDimension), sizeof(inputVectorDimension));
    outFile.write(reinterpret_cast<const char*>(&outputVectorDimension), sizeof(outputVectorDimension));
    outFile.write(reinterpret_cast<const char*>(&numberOfPresentations), sizeof(numberOfPresentations));
    outFile.write(reinterpret_cast<const char*>(&hiddenLayerCount), sizeof(hiddenLayerCount));
    outFile.write(reinterpret_cast<const char*>(&presentationNo), sizeof(presentationNo));
    outFile.write(reinterpret_cast<const char*>(&currentIteration), sizeof(currentIteration));
    outFile.write(reinterpret_cast<const char*>(&thisIterationError), sizeof(thisIterationError));
    outFile.write(reinterpret_cast<const char*>(layerInfo.data()), sizeof(LayerInfo) * layerInfo.size());

    for (const auto& inputVector : inputs) {
        outFile.write(reinterpret_cast<const char*>(inputVector.data()), sizeof(float) * inputVector.size());
    }

    for (const auto& outputVector : outputs) {
        outFile.write(reinterpret_cast<const char*>(outputVector.data()), sizeof(float) * outputVector.size());
    }

    for (int i = 0; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i]->neuronCount; ++j) {
            outFile.write(reinterpret_cast<const char*>(&layers[i]->layerNeurons[j]), sizeof(Neuron));
        }
    }

    outFile.close();
}

void Ann::readAnn() {
    std::ifstream inFile("ann.dat", std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }
    
    inFile.read(reinterpret_cast<char*>(this), sizeof(*this));

    if (!inFile) {
        std::cerr << "Error reading file or file is corrupted!" << std::endl;
        return;
    }

    inFile.close();
}

void Ann::printLayers(){
  // for(int i=0;i<hiddenLayerCount+2;i++){
  //   for(int j=0;j<layers[i]->neuronCount;j++){
  //     layers[i]->layerNeurons[j].print();
  //   }
  // }
  float outp=layers[hiddenLayerCount+1]->layerNeurons[0].fNeti;
  cout<<outp<<" ";
}
void Ann::readFile() {
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
void Ann::parseAnn(){
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
  while(thisIterationError>0.1){
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
      writeErrorsInFile();
  }
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
int main(){
  vector<LayerInfo> vec;
  LayerInfo lyr;
  for(int i=0;i<3;i++){
    lyr.activationType=unipolarSigmoidal;
    lyr.neuronCount=3;
    vec.push_back(lyr);
  }
  Ann a("./input.json","./output.json",1,1,100,vec);
  a.readFile();
  a.buildAnn();
  
  a.train();
  // Ann a;
  // a.readAnn();
  // a.printVectors();
}