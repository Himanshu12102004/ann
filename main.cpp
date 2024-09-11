#include "./ann/ann.cpp"

int main(){
  /*
LayerInfo Setup:

The `layerInfo` vector is used to configure the hidden layers of the artificial neural network. Each `LayerInfo` object in this vector specifies the properties of one hidden layer.

In this example:

- We create a `LayerInfo` object named `lyr` to represent each hidden layer.
- A loop is used to initialize three hidden layers. Each layer:
  - Uses the `unipolarSigmoidal` activation function.
  - Contains 3 neurons.

This setup means that the neural network will have three hidden layers, each with 3 neurons, all using the `unipolarSigmoidal` activation function.

Example Usage:
- The `layerInfo` vector is passed to the ANN constructor to define the architecture of the network. Each `LayerInfo` object determines the configuration of the corresponding hidden layer.
- The network architecture defined here is suitable for cases where a uniform configuration of hidden layers is desired.

Note:
- The `activationType` is set to `unipolarSigmoidal` for all hidden layers, but this can be adjusted if different activation functions are needed for different layers.
- The number of neurons in each hidden layer is set to 3, which can also be modified based on the network design requirements.
*/

  vector<LayerInfo> layerInfo;
  LayerInfo lyr;
  for(int i=0;i<1;i++){
    lyr.activationType=unipolarSigmoidal;
    lyr.neuronCount=3;
    layerInfo.push_back(lyr);
  }
/* 
This ANN constructor is used to create an artificial neural network model. 
Below is a detailed explanation of the parameters:

- inputFile: The path to the input file containing all the data (both training and testing).
- inputVectorDimension: The dimension of the input vectors (i.e., the number of input features).
- outputVectorDimension: The dimension of the output vectors (i.e., the number of output features).
- mode: The mode of the ANN, which can be one of the following:
  - initializing: Normalizes the data and splits it into training and testing datasets.
  - training: Trains the network using the training dataset.
  - testing: Tests the network's performance on the testing dataset.
  - production: Uses the trained network for making predictions on new data.
- layerInfo: A vector of LayerInfo objects that define the structure of each layer in the network. 
             Each LayerInfo object contains details like the activation type and neuron count for that layer.
- permissableError: The permissible percent error for training the network, typically used as a threshold for convergence.
- trainRatio: The ratio of the dataset used for training (e.g., 0.8 means 80% of the data is used for training).

Example Usage:
- The constructor initializes the ANN using the specified data, network architecture, 
  and training parameters. In `initializing` mode, it normalizes and splits the data before any training begins.
  The network can then be trained by calling the `start()` method.
*/
  int inputVectorDimension=1;
  int outputVectorDimension=1;
  float permissablePercentError=0.01;
  float trainRatio=1;
  Ann a("inputData/input.json",inputVectorDimension,outputVectorDimension,production,layerInfo,permissablePercentError,trainRatio);
  a.start();
  int b=int(5.9);
}