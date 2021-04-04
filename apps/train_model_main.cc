#include <core/dataset.h>
#include <core/model.h>

#include <iostream>
#include <fstream>

using naivebayes::Dataset;
using naivebayes::Image;
using naivebayes::Model;

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.

  std::string path = "/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/trainingimagesandlabels.txt";
  std::string mock = "/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/mock_data.txt";
  
  std::string model_out = "/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/model_test.json";
  
  Dataset dataset = Dataset();
  Model model = Model();

//  std::ifstream input_file(mock);
//  if (input_file.is_open()) {
//    input_file >> dataset;
//    input_file.close();
//  }
//  
//  model.Train(dataset);

//  std::ofstream output_file(model_out);
//  if (output_file.is_open()) {
//    output_file << model;
//    output_file.close();
//  }

  std::ifstream model_file(model_out);
  if (model_file.is_open()) {
    model_file >> model;
    model_file.close();
  }

  return 0;
}
