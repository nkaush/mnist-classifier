//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <cmath>

#include "core/model.h"

namespace naivebayes {

using std::vector;
using std::string;
using std::pair;

void Model::Train(const Dataset& dataset) {
  CreateClassifications(dataset);
}

void Model::CreateClassifications(const Dataset& dataset) {
  vector<char> labels = dataset.GetDistinctLabels();

  float laplace_smoothing = labels.size() * kLaplaceSmoothingFactor;
  float smoothed_dataset_size = laplace_smoothing + dataset.GetSize();

  for (char label : labels) {
    const vector<Image>& group = dataset.GetImageGroup(label);

    float smoothed_class_count = group.size() + kLaplaceSmoothingFactor;
    float likelihood = log10(smoothed_class_count / smoothed_dataset_size);

    std::map<size_t, std::vector<std::vector<float>>> feature_likelihoods;
    // TODO calculate this

    Classification classification = {likelihood, feature_likelihoods};
    classifications_.insert(pair<char, Classification>(label, classification));
  }
}

void Model::SaveToFile(string path) {}

}
