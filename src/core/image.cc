//
// Created by Neil Kaushikkar on 4/4/21.
//

#include "core/image.h"

namespace naivebayes {

using std::map;
using std::vector;
using std::string;

const map<char, Shading> Image::kPixelShadings =
    {{' ', Shading::kWhite}, {'+', Shading::kBlack}, {'#', Shading::kBlack}};

const vector<Shading> Image::kDistinctShadingEncodings =
    {Shading::kWhite, Shading::kBlack};

size_t Image::GetHeight() const {
  return pixels_.size();
}

size_t Image::GetWidth() const {
  return pixels_.at(0).size();
}

Shading Image::MapStringDigitEncodingToShading(const std::string& to_map) {
  for (const Shading& shading : kDistinctShadingEncodings) {
    string encoding_string = std::to_string(static_cast<int>(shading));
    if (to_map == encoding_string) {
      return shading;
    }
  }

  throw std::invalid_argument("The shading encoding string provided is invalid.");
}

Image::Image(std::vector<std::vector<Shading>> pixels, char label) :
  pixels_(std::move(pixels)), label_(label) {}

char Image::GetLabel() const { 
  return label_; 
}

Shading Image::GetPixel(size_t row, size_t column) const {
  return pixels_.at(row).at(column);
}

} // namespace naivebayes
