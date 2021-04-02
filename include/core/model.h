//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <map>
#include <string>

#include "core/dataset.h"

namespace naivebayes {

struct Classification {
  float class_likelihood_;
  std::map<Shading, std::vector<std::vector<float>>> shading_likelihood_;
};

class Model {
  public:
    Model() = default;

    void Train(const Dataset& dataset);

    void SaveToFile(std::string path);

  private:
    std::map<char, Classification> classifications_;

    static constexpr size_t kLaplaceSmoothingFactor = 1;

    void CreateClassifications(const Dataset& dataset);

    std::map<Shading, std::vector<std::vector<float>>>
    CalculateFeatureLikelihoods(
        const std::vector<Image>& group, size_t label_count) const;

    std::map<Shading, size_t> FindPixelShadingCounts(
        const std::vector<Image>& group, size_t row, size_t column) const;
};

} // namespace naivebayes

#endif  // NAIVE_BAYES_MODEL_H
