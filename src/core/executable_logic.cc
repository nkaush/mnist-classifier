//
// Created by Neil Kaushikkar on 4/5/21.
//

#include <iostream>
#include <fstream>

#include "core/executable_logic.h"

namespace naivebayes {

using std::vector;
using std::string;

ExecutableLogic::ExecutableLogic() : model_(Model()) {}

int ExecutableLogic::Execute(const string& train_flag, const string& load_flag, 
                             const string& save_flag, const string& test_flag,
                             bool is_test_multi_threaded) {
  // We can't allow the user to both train a model and load a model
  bool should_train = !train_flag.empty();
  bool should_load = !load_flag.empty();
  
  if (should_train && should_load) {
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
  
  // We can only test if we have a dataset and have a model loaded
  if (!test_flag.empty() && (should_train || should_load)) {
    TestModel(test_flag, is_test_multi_threaded);
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

void ExecutableLogic::TestModel(const string& dataset_path, 
                                bool is_test_multi_threaded) {

  if (!dataset_path.empty()) {
    std::ifstream input_file(dataset_path);

    if (input_file.is_open()) {
      Dataset dataset = Dataset();
      input_file >> dataset; // Add images from the training file to the dataset
      std::cout << "Testing model..." << std::endl;
      
      vector<vector<size_t>> confusion_matrix;
      if (is_test_multi_threaded) {
        confusion_matrix = model_.MultiThreadedTest(dataset);
      } else {
        confusion_matrix = model_.Test(dataset);
      }
      
      float score = Model::CalculateAccuracy(confusion_matrix);

      std::cout << "Accuracy: " << score << std::endl;
    }
  }
}

void ExecutableLogic::ValidateFilePath(const string& file_path) {
  std::ifstream located_file(file_path);

  if (!located_file.is_open()) {
    throw std::invalid_argument("The file path is not valid.");
  }

  located_file.close();
}

void ExecutableLogic::SaveConfusionMatrix() const {}

} // namespace naivebayes