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
  Dataset dataset = Dataset();

  std::ifstream input_file(path);
  if (input_file.is_open()) {
    input_file >> dataset;
    input_file.close();
  }

  Model model = Model();
  model.Train(dataset);

  /*std::ofstream output_file("path here");
  if (output_file.is_open()) {

  } else {

  }*/

  return 0;
}
