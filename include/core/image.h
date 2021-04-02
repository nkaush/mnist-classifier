//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <vector>

namespace naivebayes {

class Image {
  public:
    Image(std::vector<std::vector<size_t>> pixels_to_set, char label_to_set);

    size_t GetHeight() const;

    size_t GetWidth() const;

    char GetLabel() const;

    const std::vector<std::vector<size_t>>& GetPixels() const;

  private:
    std::vector<std::vector<size_t>> pixels_;
    char label_;
};

} // namespace naivebayes

#endif//NAIVE_BAYES_IMAGE_H
