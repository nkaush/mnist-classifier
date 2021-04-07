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
using std::map;

using Matrix = std::vector<std::vector<size_t>>;
using ThreadGroup = std::vector<std::pair<std::thread, std::future<Matrix>>>;

const string Model::kJsonSchemaLabelKey = "label";
const string Model::kJsonSchemaClassKey = "class_likelihood";
const string Model::kJsonSchemaShadingKey = "shading_likelihoods";

Model::Model(float laplace_smoothing) : laplace_smoothing_(laplace_smoothing) {}

float Model::GetClassLikelihood(char class_label) const {
  return classifications_.at(class_label).class_likelihood_;
}

float Model::GetFeatureLikelihood(char class_label, Shading shading,
                                  size_t row, size_t column) const {
  Classification specified_class = classifications_.at(class_label);
  vector<vector<float>>shading_likelihood =
      specified_class.shading_likelihoods_.at(shading);

  return shading_likelihood.at(row).at(column);
}

void Model::Train(const Dataset& dataset) {
  vector<char> labels = dataset.GetDistinctLabels();

  float laplace_smoothing = 
      static_cast<float>(labels.size())* laplace_smoothing_;
  float smoothed_dataset_size = laplace_smoothing + dataset.GetSize();

  for (char label : labels) {
    const vector<Image>& group = dataset.GetImageGroup(label);

    float smoothed_class_count = 
        static_cast<float>(group.size()) + laplace_smoothing_;

    float class_likelihood = log10(smoothed_class_count / smoothed_dataset_size);

    map<Shading, vector<vector<float>>> feature_likelihoods =
        CalculateFeatureLikelihoods(group, labels.size());

    Classification classification = {class_likelihood, feature_likelihoods};
    classifications_.insert(pair<char, Classification>(label, classification));
  }
}

map<Shading, vector<vector<float>>> Model::CalculateFeatureLikelihoods(
    const vector<Image>& class_group, size_t label_count) const {
  size_t row_count = class_group.at(0).GetHeight();
  size_t column_count = class_group.at(0).GetWidth();

  // Initialize the empty map so we can fill it by indexing as we iterate
  map<Shading, vector<vector<float>>> feature_likelihoods = 
      InitializeEmptyFeatureMap(row_count, column_count);

  float group_size_smooth_factor =
      laplace_smoothing_ * static_cast<float>(label_count);
  float smoothed_group_count = 
      group_size_smooth_factor + static_cast<float>(class_group.size());

  for (size_t row = 0; row < row_count; row++) {
    for (size_t column = 0; column < column_count; column++) {
      map<Shading, size_t> pixel_shading_counts = 
          CountPixelShadings(class_group, row, column);

      for (pair<const Shading, size_t> shading_count : pixel_shading_counts) {
        float smoothed_pixel_shading_count =
            laplace_smoothing_ + static_cast<float>(shading_count.second);

        float likelihood = 
            log10(smoothed_pixel_shading_count / smoothed_group_count);
        
        auto likelihood_iterator = feature_likelihoods.find(shading_count.first);
        likelihood_iterator->second.at(row).at(column) = likelihood;
      }
    }
  }

  return feature_likelihoods;
}

map<Shading, vector<vector<float>>> Model::InitializeEmptyFeatureMap(
    size_t row_count, size_t column_count) {
  map<Shading, vector<vector<float>>> feature_likelihoods;
  
  for (Shading shading : Image::kDistinctShadingEncodings) {
    vector<vector<float>> empty_vector;
    empty_vector.resize(row_count, vector<float>(column_count));

    feature_likelihoods.insert(
        pair<Shading, vector<vector<float>>>(shading, empty_vector));
  }
  
  return feature_likelihoods;
}

map<Shading, size_t> Model::CountPixelShadings(
    const vector<Image>& group, size_t row, size_t column) {
  // map has key shading_class and value image_count
  map<Shading, size_t> pixel_shading_counts;
  // Initialize all image_counts to 0
  for (Shading shading : Image::kDistinctShadingEncodings) {
    pixel_shading_counts.insert(pair<Shading, size_t> (shading, 0));
  }

  for (const Image& image : group) {
    Shading shading_class = image.GetPixel(row, column);

    // increment the count of the specified shading
    auto pixel_count_iterator = pixel_shading_counts.find(shading_class);
    pixel_count_iterator->second++;
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
  
  for (const json& classification : serialized_model) {
    string class_string = classification[Model::kJsonSchemaLabelKey];
    char class_label = class_string.at(0);
    
    float class_likelihood = classification[Model::kJsonSchemaClassKey];

    map<string, vector<vector<float>>> parsed_json_object = 
        classification[Model::kJsonSchemaShadingKey];
    map<Shading, vector<vector<float>>> shading_likelihood;
    
    for (const auto& json_pairs : parsed_json_object) {
      Shading shading_encoding =
          Image::MapStringDigitEncodingToShading(json_pairs.first);

      pair<Shading, vector<vector<float>>> encoded_pair(shading_encoding,
                                                        json_pairs.second);
      shading_likelihood.insert(encoded_pair);
    }

    Classification class_struct = {class_likelihood, shading_likelihood};
    model.classifications_.insert(
        pair<char, Classification>(class_label, class_struct));
  }
  
  return input;
}

char Model::Classify(const Image& image) const {
  vector<float> likelihoods;
  vector<char> labels;

  for (const auto& class_group : classifications_) {
    char label = class_group.first;
    float score = GetClassLikelihood(label);
    labels.push_back(label);

    for (size_t row = 0; row < image.GetHeight(); row++) {
      for (size_t column = 0; column < image.GetWidth(); column++) {
        Shading shading = image.GetPixel(row, column);
        score += GetFeatureLikelihood(label, shading, row, column);
      }
    }
    likelihoods.push_back(score);
  }

  size_t idx_max_score = std::distance(likelihoods.begin(),
                                       std::max_element(likelihoods.begin(),
                                                        likelihoods.end()));

  return labels.at(idx_max_score);
}

vector<vector<size_t>> Model::Test(const Dataset& dataset) const {
  map<char, size_t> label_indices;
  size_t label_index = 0;

  // Assign each char class label an index in our ConfusionMatrix
  for (const auto& classification : classifications_) {
    label_indices.insert(pair<char, size_t>(classification.first, label_index));
    label_index++;
  }

  vector<size_t> matrix_row(label_indices.size(), 0);
  Matrix confusion_matrix(label_indices.size(), matrix_row);

  size_t index = 0;
  for (char label : dataset.GetDistinctLabels()) {
    const vector<Image>& images = dataset.GetImageGroup(label);

    for (const Image& image : images) {
      if (index % 100 == 0) {
        std::cout << "Index: " << index << std::endl;
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

vector<vector<size_t>> Model::MultiThreadedTest(const Dataset& dataset) const {
  // Assign each char class label an index in our ConfusionMatrix
  map<char, size_t> label_indices;
  size_t label_index = 0;
  
  for (const auto& classification : classifications_) {
    label_indices.insert(pair<char, size_t>(classification.first, label_index));
    label_index++;
  }

  // Create threads to classify groups of Images and then aggregate results
  ThreadGroup threads = CreateTestThreads(dataset, label_indices);
  Matrix confusion_matrix = JoinTestThreads(threads, label_indices);
  
  return confusion_matrix;
}

ThreadGroup Model::CreateTestThreads(
    const Dataset& dataset, const map<char, size_t>& label_indices) const {

  ThreadGroup thread_group;
  // Adapted from https://stackoverflow.com/a/57134334
  size_t thread_index = 0;
  for (char label : dataset.GetDistinctLabels()) {
    const vector<Image>& images = dataset.GetImageGroup(label);

    promise<Matrix> thread_result;
    future<Matrix> completable_future = thread_result.get_future();

    // Adapted from https://www.reddit.com/r/cpp_questions/comments/8top49/call_to_deleted_function_when_using_threads/
    thread next_thread(&Model::TestSingleImageGroup, std::ref(*this),
                       std::move(thread_result), images, label_indices,
                       thread_index);

    thread_group.emplace_back(std::move(next_thread), 
                              std::move(completable_future));
    thread_index++;
  }
  
  return thread_group;
}

void Model::TestSingleImageGroup(
    promise<Matrix> thread_result, const vector<Image>& image_group,
    const map<char, size_t>& label_indices, size_t thread_index) const {
  vector<size_t> matrix_row(label_indices.size(), 0);
  Matrix confusion_matrix(label_indices.size(), matrix_row);

  size_t index = 0;
  for (const Image& image : image_group) {
    if (index % 20 == 0) {
      std::cout << "Thread " << thread_index << " Index: " << index << std::endl;
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

Matrix Model::JoinTestThreads(
    ThreadGroup& threads, const map<char, size_t>& label_indices) const {
  // Initialize out confusion matrix to be n-labels x n-labels of 0s
  vector<size_t> matrix_row(label_indices.size(), 0);
  Matrix confusion_matrix(label_indices.size(), matrix_row);

  // Adapted from https://stackoverflow.com/a/57134334
  // Go through each thread, get the resulting confusion matrix, and aggregate
  for (auto& thread_pair : threads) {
    // Get the thread and the promised result wrapper
    thread image_group_thread = std::move(thread_pair.first);
    future<Matrix> result = std::move(thread_pair.second);

    Matrix thread_matrix = result.get(); // Get promised result from the thread

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

float Model::CalculateAccuracy(const Matrix& confusion_matrix) {
  size_t correct = 0;
  size_t prediction_count = 0;
  
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
