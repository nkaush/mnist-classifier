//
// Created by Neil Kaushikkar on 4/4/21.
//

#include "core/image.h"

namespace naivebayes {

using std::map;
using std::vector;
using std::string;

const map<char, Shading> Image::kPixelShadings =
    {{' ', Shading::kWhite}, {'+', Shading::kGray}, {'#', Shading::kBlack}};

const vector<Shading> Image::kDistinctShadingEncodings =
    {Shading::kWhite, Shading::kBlack, Shading::kGray};

Image::Image() : pixels_(), label_('\0') {}

Image::Image(const std::vector<std::vector<Shading>>& pixels, char label) :
    pixels_(pixels), label_(label) {}

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

char Image::GetLabel() const { 
  return label_; 
}

Shading Image::GetPixel(size_t row, size_t column) const {
  return pixels_.at(row).at(column);
}

std::istream& operator>>(std::istream& input, Image& image) {
  string next_line;
  getline(input, next_line);
  
  image.label_ = next_line.at(0);
  
  while (getline(input, next_line)) {
    vector<Shading> pixel_row;
    for (char pixel : next_line) {
      pixel_row.push_back(Image::kPixelShadings.at(pixel));
    }
    image.pixels_.push_back(pixel_row);
  }
  
  return input;
}

} // namespace naivebayes
