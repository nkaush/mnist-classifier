//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <numeric>
#include <cmath>

#include "core/model.h"

namespace naivebayes {

using nlohmann::json;
using std::promise;
using std::future;
using std::thread;
using std::vector;
using std::string;
using std::pair;
using std::move;
using std::map;

using LongMatrix = vector<vector<size_t>>;
using FloatMatrix = vector<vector<float>>;
using ThreadGroup = vector<pair<thread, future<LongMatrix>>>;

const string Model::kJsonSchemaLabelKey = "label";
const string Model::kJsonSchemaClassKey = "class_likelihood";
const string Model::kJsonSchemaShadingKey = "shading_likelihoods";

const string Model::kModelTestingIndexFeedback = "Index: ";
const string Model::kModelTestingThreadFeedback = "Thread: ";

Model::Model(size_t laplace_smoothing) 
    : laplace_smoothing_(static_cast<float>(laplace_smoothing)) {}

float Model::GetClassLikelihood(char class_label) const {
  return classifications_.at(class_label).class_likelihood_;
}

float Model::GetFeatureLikelihood(char class_label, Shading shading,
                                  size_t row, size_t column) const {
  Classification specified_class = classifications_.at(class_label);
  FloatMatrix shading_likelihood =
      specified_class.shading_likelihoods_.at(shading);

  return shading_likelihood.at(row).at(column);
}

const map<char, size_t>& Model::GetLabelIndices() const {
  return label_indices_;
}

void Model::Train(const Dataset& dataset) {
  vector<char> labels = dataset.GetDistinctLabels();
  size_t label_index = 0;

  float laplace_smoothing = 
      static_cast<float>(labels.size()) * laplace_smoothing_;
  float smoothed_dataset_size = 
      laplace_smoothing + static_cast<float>(dataset.GetSize());

  for (char label : labels) {
    const vector<Image>& group = dataset.GetImageGroup(label);

    float smoothed_class_count = 
        static_cast<float>(group.size()) + laplace_smoothing_;

    float class_likelihood = log10(smoothed_class_count / smoothed_dataset_size);

    map<Shading, FloatMatrix> feature_likelihoods =
        CalculateFeatureLikelihoods(group, labels.size());

    Classification classification = {class_likelihood, feature_likelihoods};
    classifications_[label] = classification;

    // Assign each char class label an index in our confusion matrix
    label_indices_[label] = label_index;
    label_index++;
  }
}

map<Shading, FloatMatrix> Model::CalculateFeatureLikelihoods(
    const vector<Image>& class_group, size_t label_count) const {
  size_t row_count = class_group.at(0).GetHeight();
  size_t column_count = class_group.at(0).GetWidth();

  // Initialize the empty map so we can fill it by indexing as we iterate
  map<Shading, FloatMatrix> feature_likelihoods = 
      InitializeEmptyFeatureMap(row_count, column_count);
  
  float group_size_smooth_factor =
      laplace_smoothing_ * static_cast<float>(label_count);
  float smoothed_group_count = 
      group_size_smooth_factor + static_cast<float>(class_group.size());

  // Go through each pixel and calculate each feature likelihood
  for (size_t row = 0; row < row_count; row++) {
    for (size_t column = 0; column < column_count; column++) {
      map<Shading, size_t> pixel_shading_counts = 
          CountPixelShadings(class_group, row, column);

      // Go through all Shading types and calculate likelihood for each
      for (const auto& shading_count : pixel_shading_counts) {
        float smoothed_pixel_shading_count =
            laplace_smoothing_ + static_cast<float>(shading_count.second);

        float likelihood = 
            log10(smoothed_pixel_shading_count / smoothed_group_count);
        
        feature_likelihoods.at(shading_count.first).at(row).at(column) = 
            likelihood;
      }
    }
  }

  return feature_likelihoods;
}

map<Shading, FloatMatrix> Model::InitializeEmptyFeatureMap(
    size_t row_count, size_t column_count) {
  map<Shading, FloatMatrix> feature_likelihoods;
  
  // Fill in the feature map so we explicitly say that there can be 0 images
  for (const Shading& shading : Image::kDistinctShadingEncodings) {
    FloatMatrix empty_vector(row_count, vector<float>(column_count));
    feature_likelihoods[shading] = empty_vector;
  }
  
  return feature_likelihoods;
}

map<Shading, size_t> Model::CountPixelShadings(
    const vector<Image>& group, size_t row, size_t column) {
  // map has key shading_class and value image_count
  map<Shading, size_t> pixel_shading_counts;
  
  // Initialize all image_counts to 0
  for (const Shading& shading : Image::kDistinctShadingEncodings) {
    pixel_shading_counts[shading] = 0;
  }

  for (const Image& image : group) {
    Shading shading_class = image.GetPixel(row, column);

    // increment the count of the specified shading
    pixel_shading_counts.at(shading_class)++;
  }

  return pixel_shading_counts;
}

std::ostream& operator<<(std::ostream& output, const Model& model) {
  json serialized_model = json::array();
  
  // Go through each label/Classification struct pair so we can serialize them
  for (const auto& classification : model.classifications_) {
    json classification_object;
    // convert the char to a string with that 1 char
    classification_object[Model::kJsonSchemaLabelKey] = 
        std::string(1, classification.first);
 
    Classification class_struct = classification.second;
    classification_object[Model::kJsonSchemaClassKey] = 
        class_struct.class_likelihood_;
    
    json shading_likelihoods;
    // Go through the map in the Classification struct and serialize it
    for (const auto& shading_likelihood : class_struct.shading_likelihoods_) {
      // cast the encoding of the Shading enum to an int, then convert to string
      int shading_encoding = static_cast<int>(shading_likelihood.first);
      string shading_key = std::to_string(shading_encoding);
      
      shading_likelihoods[shading_key] = shading_likelihood.second;
    }
    
    classification_object[Model::kJsonSchemaShadingKey] = shading_likelihoods;
    serialized_model.push_back(classification_object);
  }
  
  output << serialized_model.dump(Model::kJsonSchemaSpacing) << std::endl;
  return output;
}

std::istream& operator>>(std::istream& input, Model& model) {
  json serialized_model;
  input >> serialized_model;
  
  // Go through each classification json object to deserialize them
  for (const json& classification : serialized_model) {
    // Find the serialized class label and class likelihood
    string class_string = classification[Model::kJsonSchemaLabelKey];
    char class_label = class_string.at(0);
    float class_likelihood = classification[Model::kJsonSchemaClassKey];

    map<string, FloatMatrix> parsed_json_object = 
        classification[Model::kJsonSchemaShadingKey];
    
    map<Shading, FloatMatrix> shading_likelihood;
    // Go through each 2D-array containing likelihoods for each Shading type
    for (const auto& json_pairs : parsed_json_object) {
      Shading shading_encoding =
          Image::MapStringDigitEncodingToShading(json_pairs.first);
      
      shading_likelihood[shading_encoding] = json_pairs.second;
    }

    Classification class_struct = {class_likelihood, shading_likelihood};
    model.classifications_[class_label] = class_struct;
  }
  
  return input;
}

char Model::Classify(const Image& image) const {
  char most_likely_label = Image::kDefaultLabel;
  // set to an arbitrary number, will be reset after 1st likelihood calculation
  float max_likelihood = 0; 
  
  // Go through each class label to check the likelihood of being that label
  for (const auto& class_group : classifications_) {
    char label = class_group.first;
    float score = CalculateLikelihoodScore(label, image);
    
    // Update the prediction if we hit the first class or have a higher score
    if (score > max_likelihood || most_likely_label == Image::kDefaultLabel) {
      max_likelihood = score;
      most_likely_label = label;
    }
  }
  
  return most_likely_label;
}

float Model::CalculateLikelihoodScore(char label, const Image& image) const { 
  float score = GetClassLikelihood(label); 

  // Go through each pixel to retrieve likelihoods of the shading of the pixel
  for (size_t row = 0; row < image.GetHeight(); row++) {
    for (size_t column = 0; column < image.GetWidth(); column++) {
      Shading shading = image.GetPixel(row, column);
      score += GetFeatureLikelihood(label, shading, row, column);
    }
  }
  
  return score;
}

LongMatrix Model::Test(const Dataset& dataset, bool is_printing_verbose) const {
  map<char, size_t> label_indices = GetLabelIndices();
  
  vector<size_t> matrix_row(label_indices.size(), 0);
  LongMatrix confusion_matrix(label_indices.size(), matrix_row);

  size_t index = 0;
  // Go through each subset of images in the dataset
  for (char label : dataset.GetDistinctLabels()) {
    const vector<Image>& images = dataset.GetImageGroup(label);

    // Go through each image in the subset and try to predict its label
    for (const Image& image : images) {
      if (is_printing_verbose && index % kLinearTestingFeedbackRate == 0) {
        std::cout << kModelTestingIndexFeedback << index << std::endl;
      }
      char predicted = Classify(image);

      size_t row = label_indices.at(image.GetLabel());
      size_t column = label_indices.at(predicted);

      confusion_matrix.at(row).at(column)++;
      index++;
    }
  }

  return confusion_matrix;
}

LongMatrix Model::MultiThreadedTest(const Dataset& dataset, 
                                    bool is_printing_verbose) const {
  map<char, size_t> label_indices = GetLabelIndices();

  // Create threads to classify groups of Images and then aggregate results
  ThreadGroup threads = CreateTestThreads(dataset, label_indices, 
                                          is_printing_verbose);
  LongMatrix confusion_matrix = JoinTestThreads(threads, label_indices);
  
  return confusion_matrix;
}

ThreadGroup Model::CreateTestThreads(const Dataset& dataset, 
                                     const map<char, size_t>& label_indices, 
                                     bool is_printing_verbose) const {
  ThreadGroup thread_group;
  size_t thread_index = 0;
  
  // Go through each classification group and create AT LEAST 1 thread for each
  for (char label : dataset.GetDistinctLabels()) {
    const vector<Image>& images = dataset.GetImageGroup(label);
    // Account for truncating by adding 1 to ensure extra groups are not created
    size_t batch_size = (images.size() / kThreadsPerGroup) + 1; 

    size_t current_group_index = 0;
    // Split vector of images into groups of size batch_size
    while (current_group_index < images.size()) {
      // Get a sub-vector of the group of images with `batch_size` total images
      auto begin_iterator = images.begin() + current_group_index;
      current_group_index += batch_size;
      auto end_iterator = images.begin() + current_group_index;
      
      if (end_iterator > images.end()) {
        end_iterator = images.end();
      }
      
      // Adapted from https://stackoverflow.com/a/57134334
      vector<Image> sub_group(begin_iterator, end_iterator);
      promise<LongMatrix> thread_result;
      future<LongMatrix> completable_future = thread_result.get_future();
      
      // Adapted from https://www.reddit.com/r/cpp_questions/comments/8top49/call_to_deleted_function_when_using_threads/
      thread next_thread(&Model::TestImageGroup, std::ref(*this),
                         std::move(thread_result), sub_group, 
                         label_indices, thread_index, is_printing_verbose);
      // Create a thread for each group of images
      thread_group.emplace_back(move(next_thread), move(completable_future));
      thread_index++;
    }
  }

  return thread_group;
}

void Model::TestImageGroup(
    promise<LongMatrix> thread_result, const vector<Image>& image_group,
    const map<char, size_t>& label_indices, size_t thread_index, 
    bool is_printing_verbose) const {
  // Initialize an empty confusion matrix we fill with results of this thread
  vector<size_t> matrix_row(label_indices.size(), 0);
  LongMatrix confusion_matrix(label_indices.size(), matrix_row);

  size_t index = 0;
  for (const Image& image : image_group) {
    // Only print label index every `kThreadedTestingFeedbackRate` predictions
    if (is_printing_verbose && index % kThreadedTestingFeedbackRate == 0) {
      std::cout << kModelTestingThreadFeedback << thread_index << "\t";
      std::cout << kModelTestingIndexFeedback << index << std::endl;
    }
    
    char predicted_label = Classify(image);

    // Find the indices assigned to both the predicted and actual labels
    size_t row = label_indices.at(image.GetLabel());
    size_t column = label_indices.at(predicted_label);

    confusion_matrix.at(row).at(column)++;
    index++;
  }

  thread_result.set_value(confusion_matrix);
}

LongMatrix Model::JoinTestThreads(
    ThreadGroup& threads, const map<char, size_t>& label_indices) const {
  // Initialize out confusion matrix to be n-labels x n-labels of 0s
  vector<size_t> matrix_row(label_indices.size(), 0);
  LongMatrix confusion_matrix(label_indices.size(), matrix_row);
  
  // Go through each thread, get the resulting confusion matrix, and aggregate
  for (pair<thread, future<LongMatrix>>& thread_pair : threads) {
    // Adapted from https://stackoverflow.com/a/57134334
    thread image_group_thread = std::move(thread_pair.first);
    future<LongMatrix> result = std::move(thread_pair.second);

    LongMatrix thread_matrix = result.get(); // retrieve result from promise

    // Aggregate the results
    for (size_t row = 0; row < confusion_matrix.size(); row++) {
      for (size_t col = 0; col < confusion_matrix.at(0).size(); col++) {
        confusion_matrix.at(row).at(col) += thread_matrix.at(row).at(col);
      }
    }

    image_group_thread.join(); // Close the thread
  }
  
  return confusion_matrix;
}

float Model::CalculateAccuracy(const LongMatrix& confusion_matrix) {
  size_t correct = 0;
  size_t prediction_count = 0;
  
  // Go through the confusion matrix to find the number of correct predictions
  for (size_t row = 0; row < confusion_matrix.size(); row++) {
    // The diagonal of the matrix is when actual label == predicted label
    correct += confusion_matrix.at(row).at(row);
    
    for (size_t value : confusion_matrix.at(row)) {
      prediction_count += value;
    }
  }

  return static_cast<float>(correct) / static_cast<float>(prediction_count);
}

} // namespace naivebayes
