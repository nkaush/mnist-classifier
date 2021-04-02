//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <nlohmann/json.hpp>
#include <string>
#include <map>

#include "core/dataset.h"

namespace naivebayes {

struct Classification {
  float class_likelihood_;
  std::map<Shading, std::vector<std::vector<float>>> shading_likelihoods_;
};

class Model {
  public:
    Model() = default;

    void Train(const Dataset& dataset);
    
    float GetClassLikelihood(char class_label) const;
    
    float GetFeatureLikelihood(char class_label, Shading shading, 
                               size_t row, size_t column) const;

    friend std::ostream &operator<<(std::ostream &output, const Model& model);

    friend std::istream &operator>>(std::istream &input, Model& model);

  private:
    std::map<char, Classification> classifications_;

    static constexpr size_t kLaplaceSmoothingFactor = 1;
    
    std::map<Shading, std::vector<std::vector<float>>>
    CalculateFeatureLikelihoods(
        const std::vector<Image>& class_group, size_t label_count) const;

    std::map<Shading, size_t> FindPixelShadingCounts(
        const std::vector<Image>& group, size_t row, size_t column) const;
};

} // namespace naivebayes

#endif  // NAIVE_BAYES_MODEL_H
