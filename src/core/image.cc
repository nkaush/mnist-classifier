//
// Created by Neil Kaushikkar on 4/1/21.
//

#include "core/image.h"

#include <utility>

namespace naivebayes {

using std::vector;

Image::Image(vector<vector<size_t>> pixels_to_set, char label_to_set) :
  pixels_(std::move(pixels_to_set)), label_(label_to_set) {}

size_t Image::GetWidth() const {
  // Assume that image rows are uniform (we check in parsing function)
  return pixels_.at(0).size();
}

size_t Image::GetHeight() const {
  return pixels_.size();
}

char Image::GetLabel() const {
  return label_;
}

const std::vector<std::vector<size_t>>& Image::GetPixels() const {
  return pixels_;
}

}