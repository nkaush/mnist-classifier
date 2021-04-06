//
// Created by Neil Kaushikkar on 4/5/21.
//

#include <iostream>
#include <fstream>

#include "core/executable_logic.h"

namespace naivebayes {

using std::string;

ExecutableLogic::ExecutableLogic() : model_(Model()) {}

int ExecutableLogic::Execute(const string& train_flag, const string& load_flag, 
                             const string& save_flag) {
  // We can't allow the user to both train a model and load a model
  if (!train_flag.empty() && !load_flag.empty()) {
    std::cout << "You must either train a model or load a model, not both!";
    std::cout << std::endl;

    return EXIT_FAILURE; 
  } else if (!train_flag.empty()) {
    TrainModel(train_flag);
  } else if (!load_flag.empty()) {
    LoadModel(load_flag);
  }

  if (!save_flag.empty()) {
    SaveModel(save_flag);
  }
  
  return EXIT_SUCCESS;
}

void ExecutableLogic::SaveModel(const string& file_path) const {
  std::ofstream output_file(file_path);

  if (output_file.is_open()) {
    std::cout << "Saving model...";
    output_file << model_;  // Serialize the model and save to the given file
    std::cout << "done." << std::endl;
  }
}

void ExecutableLogic::LoadModel(const string& model_path) {
  std::ifstream model_file(model_path);

  if (model_file.is_open()) {
    std::cout << "Loading model...";
    model_file >> model_;  // Deserialize the model and load it in the stack
    std::cout << "done." << std::endl;
  }
}

void ExecutableLogic::TrainModel(const string& dataset_path) {
  std::ifstream input_file(dataset_path);

  if (input_file.is_open()) {
    Dataset dataset = Dataset();
    input_file >> dataset;  // Add images from the training file to the dataset
    std::cout << "Training model...";

    model_.Train(dataset);
    std::cout << "done." << std::endl;
  }
}

void ExecutableLogic::ValidateFilePath(const string& file_path) {
  std::ifstream located_file(file_path);

  if (!located_file.is_open()) {
    throw std::invalid_argument("The file path is not valid.");
  }

  located_file.close();
}

} // namespace naivebayes