#include "../neuron/neuron.cpp"
#include "../utils/json.hpp"
using json = nlohmann::json;
class Layer{
  public:
  vector<Neuron> layerNeurons;
  Layer(int neuronCount,int layerNumber);
  int neuronCount;
  int layerNumber;
  float learningConstant=0.9;
  Vector passBackError();
  virtual  void computeDelE(Vector passedBackError);
  virtual void computeDelW(Vector previousLayerOutput);
  virtual void buildLayer(int lastLayerNeuronCount,ActivationTypes activationType=unipolarSigmoidal )=0;
  void buildLayer(json layerInfo,ActivationTypes activationType=unipolarSigmoidal);
  json parseLayer();
};
Layer::Layer(int neuronCount,int layerNumber):neuronCount(neuronCount),layerNumber(layerNumber){
  this->layerNeurons=vector<Neuron>(neuronCount);
}

Vector Layer::passBackError(){
  int lengthOfPassBack=layerNeurons[0].weightDimension;
  Vector passback(lengthOfPassBack);
  for(int i=0;i<lengthOfPassBack;i++){
    for(int j=0;j<neuronCount;j++){
      passback[i]+=layerNeurons[j].delE*layerNeurons[j].weight[i];
    }
  }
  return passback;
}
void Layer::computeDelE(Vector passedBackError){
  for(int i=0;i<neuronCount;i++){
  layerNeurons[i].delE=passedBackError[i]*layerNeurons[i].calcFDashNeti();
  layerNeurons[i].delB=learningConstant*layerNeurons[i].delE;
  }
}
void Layer::computeDelW(Vector previousLayerOutput){
for(int i=0;i<neuronCount;i++){
  layerNeurons[i].delW=previousLayerOutput*layerNeurons[i].delE*learningConstant;
}
}
json Layer::parseLayer(){
    json layerJson;
        layerJson["neuronCount"] = neuronCount;
        layerJson["layerNumber"] =layerNumber;
        layerJson["neurons"] = json::array();
        for (auto& neuron : layerNeurons) {
            json neuronJson=neuron.parseNeuron();
            layerJson["neurons"].push_back(neuronJson);
        }
    return layerJson;
}
void Layer::buildLayer(json layerSpecs,ActivationTypes activationType){
   layerNumber=layerSpecs["layerNumber"];
   neuronCount=layerSpecs["neuronCount"];
   layerNeurons=vector<Neuron>(neuronCount);
   if(layerNumber==0){
    for(int i=0;i<neuronCount;i++){
  layerNeurons[i]=(Neuron(nothing,vector<float>(1,1)));
  }
   }
   else{
    for(int i=0;i<neuronCount;i++){
    vector<float> weight = layerSpecs["neurons"][i]["weight"].get<std::vector<float>>();
    float bias=layerSpecs["neurons"][i]["bias"];
    Vector weightWrapped=Vector(weight);
    layerNeurons[i]=Neuron(activationType,weightWrapped,bias);
    }
   }
}
