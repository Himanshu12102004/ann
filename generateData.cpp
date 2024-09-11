#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "utils/json.hpp"
#include "utils/randomNumberGenerator.h"
using json = nlohmann::json;
using namespace  std;
struct DataPoint {
    vector<double> inputs;
    vector<double> outputs;
};

int main() {
    vector<DataPoint> data;
    for (int i = 0; i < 100; ++i) {
        double x = generateRandomFloat(0,1);
        double p =x*x;
        DataPoint point;
        point.inputs = {x};
        point.outputs = {p};
        data.push_back(point);
    }

    json j_data = json::array();
    for (const auto& point : data) {
        j_data.push_back({{"inputs", point.inputs}, {"outputs", point.outputs}});
    }

    ofstream file("inputData/input.json");
    file << j_data.dump(4); 
     cout<<"Dataset Generated Successfully\n";
    return 0;
}
