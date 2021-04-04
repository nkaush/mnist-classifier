#include <gflags/gflags.h>

#include <core/dataset.h>
#include <core/model.h>

#include <iostream>
#include <fstream>

// Define the command line arguments and set the default value to empty strings
DEFINE_string(train, "", "The file path to the dataset to train the model on.");
DEFINE_string(save, "", "The file path to save the trained model to.");
DEFINE_string(load, "", "The file path to load a pre-trained model from.");

using naivebayes::Dataset;
using naivebayes::Image;
using naivebayes::Model;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Model model = Model();
  
  // We can't allow the user to both train a model and load a model 
  if (!FLAGS_train.empty() && !FLAGS_load.empty()) {
    std::cout << "You must either train a model or load a model, not both!";
    std::cout << std::endl;
    
    return EXIT_FAILURE;
  } else if (!FLAGS_train.empty()) {
    std::ifstream input_file(FLAGS_train);
    
    if (input_file.is_open()) {
      Dataset dataset = Dataset(); 
      input_file >> dataset; // Add images from the training file to the dataset
      std::cout << "Training model...";
      
      model.Train(dataset);
      std::cout << "done." << std::endl;
    }
  } else if (!FLAGS_load.empty()) {
    std::ifstream model_file(FLAGS_load);
    
    if (model_file.is_open()) {
      std::cout << "Loading model...";
      model_file >> model; // Deserialize the model and load it in the stack
      std::cout << "done." << std::endl;
    }
  }

  if (!FLAGS_save.empty()) {
    std::ofstream output_file(FLAGS_save);
    
    if (output_file.is_open()) {
      std::cout << "Saving model...";
      output_file << model; // Serialize the model and save to the given file
      std::cout << "done." << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
