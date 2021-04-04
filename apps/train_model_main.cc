#include <gflags/gflags.h>

#include <core/dataset.h>
#include <core/model.h>

#include <iostream>
#include <fstream>

DEFINE_string(train, "", "The file path to the dataset to train the model on");
DEFINE_string(save, "", "The file path to save the trained model to");
DEFINE_string(load, "", "The file path to load a pre-trained model from");

using naivebayes::Dataset;
using naivebayes::Image;
using naivebayes::Model;

// TODO: You may want to change main's signature to take in argc and argv
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Model model = Model();
  Dataset dataset = Dataset();
  
  if (!FLAGS_train.empty() && !FLAGS_load.empty()) {
    std::cout << "You must either train a model or load a model, not both!";
    std::cout << std::endl;
    
    return EXIT_FAILURE;
  } else if (!FLAGS_train.empty()) {
    std::ifstream input_file(FLAGS_train);
    
    if (input_file.is_open()) {
      input_file >> dataset;
      std::cout << "Training model...";
      model.Train(dataset);
      std::cout << "done." << std::endl;
    }
  } else if (!FLAGS_load.empty()) {
    std::ifstream model_file(FLAGS_load);
    
    if (model_file.is_open()) {
      std::cout << "Loading model...";
      model_file >> model;
      std::cout << "done." << std::endl;
    }
  }
  
  if (!FLAGS_save.empty()) {
    std::ofstream output_file(FLAGS_save);
    
    if (output_file.is_open()) {
      std::cout << "Saving model...";
      output_file << model;
      std::cout << "done." << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
