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
const string ExecutableLogic::kConfusionMatrixRowLabel = "Actual";
const string ExecutableLogic::kConfusionMatrixLabelIndicator = "Label";

const string ExecutableLogic::kFinishedMessage = "done.";
const string ExecutableLogic::kFailedMessage = "failed.";

ExecutableLogic::ExecutableLogic(size_t laplace_factor) 
    : model_(Model(laplace_factor)) {}

int ExecutableLogic::Execute(const string& train_flag, const string& load_flag, 
                             const string& save_flag, const string& test_flag,
                             const string& confusion_flag, 
                             bool is_printing_verbose) {
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
    TestModel(test_flag, confusion_flag, is_printing_verbose);
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
                                const string& confusion_csv_path,
                                bool is_printing_verbose) const {
  std::ifstream input_file(dataset_path);

  std::cout << kTestingModelMessage << std::endl;
  if (input_file.is_open()) {
    Dataset dataset = Dataset();
    input_file >> dataset; // Add images from the training file to the dataset
    
    // Test the model via the method defined with command line flags
    vector<vector<size_t>> confusion_matrix 
        = model_.Test(dataset, is_printing_verbose);
    
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
    size_t middle_index = matrix.size() / 2; // middle is half the size...
    size_t count_after_middle = matrix.size() - middle_index;
    
    // Add one extra comma to account for the offset row labels
    output_file << string(middle_index + 1, kCsvElementDelimiter);
    output_file << kConfusionMatrixColumnLabel;
    output_file << string(count_after_middle, kCsvElementDelimiter) << std::endl;
    output_file << kCsvElementDelimiter << kConfusionMatrixLabelIndicator;
    
    // Subtract 1 to account for buffer of the column label
    WriteConfusionMatrixCounts(output_file, matrix, middle_index - 1);
    
    output_file.close();
    std::cout << kFinishedMessage << std::endl;
  } else {
    std::cout << kFailedMessage << std::endl;
  }
}

void ExecutableLogic::WriteConfusionMatrixCounts(
    std::ofstream& output_file, const vector<vector<size_t>>& matrix, 
    size_t middle_index) const {
  // Insert corresponding label into stored index and add to csv column labels
  map<char, size_t> label_indices = model_.GetLabelIndices();
  vector<char> row_labels = vector<char>(label_indices.size());
  
  for (const auto& label_pairs : label_indices) {
    char label = label_pairs.first;
    row_labels.at(label_pairs.second) = label;
    output_file << kCsvElementDelimiter << label;
  }
  output_file << std::endl;
  
  // Write each prediction count in each row to the file
  for (size_t row_idx = 0; row_idx < matrix.size(); row_idx++) {
    if (row_idx == middle_index) { // when at the middle, print the row label
      output_file << kConfusionMatrixRowLabel;
    }

    output_file << kCsvElementDelimiter << row_labels.at(row_idx);

    // Write each prediction count
    for (size_t count : matrix.at(row_idx)) {
      output_file << kCsvElementDelimiter << count;
    }

    output_file << std::endl;
  }
}

} // namespace naivebayes