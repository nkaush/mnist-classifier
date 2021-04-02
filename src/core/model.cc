//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <numeric>
#include <cmath>

#include "core/model.h"

namespace naivebayes {

using nlohmann::json;
using std::vector;
using std::string;
using std::pair;
using std::map;

void Model::Train(const Dataset& dataset) {
  vector<char> labels = dataset.GetDistinctLabels();

  float laplace_smoothing = labels.size() * kLaplaceSmoothingFactor;
  float smoothed_dataset_size = laplace_smoothing + dataset.GetSize();

  for (char label : labels) {
    const vector<Image>& group = dataset.GetImageGroup(label);

    float smoothed_class_count = (float) group.size() + kLaplaceSmoothingFactor;
    float likelihood = log10(smoothed_class_count / smoothed_dataset_size);

    map<Shading, vector<vector<float>>> feature_likelihoods =
        CalculateFeatureLikelihoods(group, labels.size());

    Classification classification = {likelihood, feature_likelihoods};
    classifications_.insert(pair<char, Classification>(label, classification));
  }
}

float Model::GetClassLikelihood(char class_label) const { 
  return classifications_.at(class_label).class_likelihood_; 
}

float Model::GetFeatureLikelihood(char class_label, Shading shading,
                                  size_t row, size_t column) const {
  auto specified_class = classifications_.at(class_label);
  auto shading_likelihood = specified_class.shading_likelihoods_.at(shading);
  
  return shading_likelihood.at(row).at(column);
}

map<Shading, vector<vector<float>>> Model::CalculateFeatureLikelihoods(
    const vector<Image>& group, size_t label_count) const {
  size_t row_count = group.at(0).GetHeight();
  size_t column_count = group.at(0).GetWidth();

  map<Shading, vector<vector<float>>> feature_likelihoods;
  // Initialize the empty map so we can fill it as we go without index errors
  for (Shading shading : Dataset::kDistinctShadingEncodings) {
    vector<vector<float>> empty_vector;
    empty_vector.resize(row_count, vector<float>(column_count));

    feature_likelihoods.insert(
        pair<Shading, vector<vector<float>>>(shading, empty_vector));
  }

  float group_size_smooth_factor = kLaplaceSmoothingFactor * (float) label_count;
  float smoothed_group_count = group_size_smooth_factor + (float) group.size();

  for (size_t row = 0; row < row_count; row++) {
    for (size_t column = 0; column < column_count; column++) {
      map<Shading, size_t> pixel_shading_counts =
          FindPixelShadingCounts(group, row, column);

      for (auto& shading_count : pixel_shading_counts) {
        float smoothed_pixel_shading_count =
            kLaplaceSmoothingFactor + (float) shading_count.second;

        float likelihood =
            log10(smoothed_pixel_shading_count / smoothed_group_count);
        
        auto map_pointer = feature_likelihoods.find(shading_count.first);
        map_pointer->second.at(row).at(column) = likelihood;
      }
    }
  }

  return feature_likelihoods;
}

map<Shading, size_t> Model::FindPixelShadingCounts(
    const vector<Image>& group, size_t row, size_t column) const {
  // map has key shading_class and value image_count
  map<Shading, size_t> pixel_shading_counts;

  // Initialize all counts to 0
  for (Shading shading : Dataset::kDistinctShadingEncodings) {
    pixel_shading_counts.insert(pair<Shading, size_t> (shading, 0));
  }

  for (const Image& image : group) {
    Shading shading_class = image.pixels_.at(row).at(column);

    // increment the count of the specified shading
    auto map_pointer = pixel_shading_counts.find(shading_class);
    map_pointer->second++;
  }

  return pixel_shading_counts;
}

std::ostream& operator<<(std::ostream& output, const Model& model) {
  json serialized_model = "";
  
  output << serialized_model.dump(4) << std::endl;
  return output;
}

std::istream& operator>>(std::istream& input, Model& model) {
  json serialized_model;
  input >> serialized_model;
  
  return input;
}

} // namespace naivebayes
