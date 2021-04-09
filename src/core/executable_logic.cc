//
// Created by Neil Kaushikkar on 4/5/21.
//

#include <iostream>
#include <fstream>

#include "core/executable_logic.h"

namespace naivebayes {

using std::vector;
using std::string;
using std::pair;
using std::map;

const string ExecutableLogic::kModelAccuracyMessage = "Accuracy: ";
const string ExecutableLogic::kTestingModelMessage = "Testing model...";

const string ExecutableLogic::kTrainingModelMessage = "Training model...";

const string ExecutableLogic::kLoadingModelMessage = "Loading model...";
const string ExecutableLogic::kLoadingConflictMessage = 
    "You must either train a model or load a model, not both!";

const string ExecutableLogic::kSavingModelMessage = "Saving model...";
const string ExecutableLogic::kSavingConfusionMatrixMessage = 
    "Saving confusion matrix...";
const string ExecutableLogic::kConfusionMatrixColumnLabel = "Predicted";

const string ExecutableLogic::kFinishedMessage = "done.";
const string ExecutableLogic::kFailedMessage = "done.";

ExecutableLogic::ExecutableLogic(size_t laplace_factor) 
    : model_(Model(laplace_factor)) {}

int ExecutableLogic::Execute(const string& train_flag, const string& load_flag, 
                             const string& save_flag, const string& test_flag,
                             const string& confusion_flag, 
                             bool is_test_multi_threaded) {
  // We can't allow the user to both train a model and load a model
  bool should_train = !train_flag.empty();
  bool should_load = !load_flag.empty();
  
  if (should_train && should_load) {
    std::cout << kLoadingConflictMessage;
    std::cout << std::endl;

    return EXIT_FAILURE; 
  } else if (should_train) {
    TrainModel(train_flag);
  } else if (should_load) {
    LoadModel(load_flag);
  }

  if (!save_flag.empty()) {
    SaveModel(save_flag);
  }
  
  // We can only test if we have a dataset and have a model loaded
  if (!test_flag.empty() && (should_train || should_load)) {
    TestModel(test_flag, is_test_multi_threaded, confusion_flag);
  }
  
  return EXIT_SUCCESS;
}

void ExecutableLogic::SaveModel(const string& file_path) const {
  std::ofstream output_file(file_path);

  std::cout << kSavingModelMessage;
  if (output_file.is_open()) {
    output_file << model_;  // Serialize the model and save to the given file
    std::cout << kFinishedMessage << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

void ExecutableLogic::LoadModel(const string& model_path) {
  std::ifstream model_file(model_path);

  std::cout << kLoadingModelMessage;
  if (model_file.is_open()) {
    model_file >> model_;  // Deserialize the model and load it in the stack
    std::cout << kFinishedMessage << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

void ExecutableLogic::TrainModel(const string& dataset_path) {
  std::ifstream input_file(dataset_path);

  std::cout << kTrainingModelMessage;
  if (input_file.is_open()) {
    Dataset dataset = Dataset();
    input_file >> dataset;  // Add images from the training file to the dataset
    
    model_.Train(dataset);
    std::cout << kFinishedMessage << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

void ExecutableLogic::TestModel(const string& dataset_path, 
                                bool is_test_multi_threaded,
                                const string& confusion_csv_path) const {
  std::ifstream input_file(dataset_path);

  std::cout << kTestingModelMessage << std::endl;
  if (input_file.is_open()) {
    Dataset dataset = Dataset();
    input_file >> dataset; // Add images from the training file to the dataset
    
    // Test the model via the method defined with command line flags
    vector<vector<size_t>> confusion_matrix;
    if (is_test_multi_threaded) {
      confusion_matrix = model_.MultiThreadedTest(dataset);
    } else {
      confusion_matrix = model_.Test(dataset);
    }
    
    // Save the confusion matrix, if specified
    if (!confusion_csv_path.empty()) {
      SaveConfusionMatrix(confusion_csv_path, confusion_matrix);
    }
    
    float score = Model::CalculateAccuracy(confusion_matrix);
    std::cout << kModelAccuracyMessage << score << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

void ExecutableLogic::SaveConfusionMatrix(
    const string& save_path, const vector<vector<size_t>>& matrix) const {
  std::ofstream output_file(save_path);

  std::cout << kSavingConfusionMatrixMessage;
  if (output_file.is_open()) {
    map<char, size_t> label_indices = model_.GetLabelIndices();
    
    size_t middle_index = matrix.size() / 2; // middle is half the size...
    // We should not add a comma AFTER the last element, so subtract 1
    size_t count_after_middle = matrix.size() - middle_index - 1;
    
    // Add one extra comma to account for the offset row labels
    output_file << string(middle_index + 1, kCsvElementDelimiter);
    output_file << kConfusionMatrixColumnLabel;
    output_file << string(count_after_middle, kCsvElementDelimiter) << std::endl;
    
    // Insert corresponding label into stored index and add to csv column labels
    vector<char> row_labels = vector<char>(label_indices.size());
    for (const auto& label_pairs : label_indices) {
      char label = label_pairs.first;
      row_labels.at(label_pairs.second) = label;
      output_file << kCsvElementDelimiter << label;
    }
    output_file << std::endl;
    
    // Write each row to the file
    for (size_t row_idx = 0; row_idx < matrix.size(); row_idx++) {
      output_file << row_labels.at(row_idx);
      for (size_t count : matrix.at(row_idx)) {
        output_file << kCsvElementDelimiter << count;
      }
      output_file << std::endl;
    }
    
    output_file.close();
    std::cout << kFinishedMessage << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

} // namespace naivebayes