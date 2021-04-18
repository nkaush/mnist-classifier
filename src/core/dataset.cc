//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <iostream>
#include <sstream>

#include "core/dataset.h"
#include "core/image.h"

namespace naivebayes {

using std::stringstream;
using std::istream;
using std::vector;
using std::string;
using std::map;

Dataset::Dataset() : size_(0) {}

const vector<Image>& Dataset::GetImageGroup(char class_label) const {
  return class_groups_.at(class_label);
}

size_t Dataset::GetSize() const {
  return size_;
}

std::vector<char> Dataset::GetDistinctLabels() const {
  vector<char> labels;

  for (const auto& group : class_groups_) {
    labels.push_back(group.first);
  }

  return labels;
}

std::istream& operator>>(istream& input, Dataset& dataset) {
  Image first_image = dataset.ParseFirstImage(input);

  size_t image_height = first_image.GetHeight();
  size_t image_width = first_image.GetWidth();
  char first_image_label = first_image.GetLabel();
  
  // Get the first image so we know dimensions to search for other images
  dataset.class_groups_[first_image_label].push_back(first_image);
  dataset.size_++;

  string current_label;
  while (!input.eof() && getline(input, current_label)) {
    // continue while there is a label
    string next_line;

    if (current_label.empty()) {
      throw std::invalid_argument("Image is missing a label.");
    }
    
    char label = current_label.at(0);  // assume, for now the label is 1st char
    stringstream image_lines;
    image_lines << label << Dataset::kFileLineDelimiter;

    // aggregate all the lines in the image, assuming all images are same size
    for (size_t line_index = 0; line_index < image_height; line_index++) {
      getline(input, next_line);
      // Only need to check width because if height was off line would be a label
      if (next_line.size() != image_width) {
        // label lengths are also different than line lengths
        throw std::invalid_argument("The images are not of uniform size");
      }
      image_lines << next_line << Dataset::kFileLineDelimiter;
    }
    
    Image image;
    image_lines >> image;
    
    dataset.class_groups_[label].push_back(image);
    dataset.size_++;
  }

  return input;
}

Image Dataset::ParseFirstImage(istream& input) const {
  string label_string;
  getline(input, label_string);

  // If the file is empty, the first line will be empty!
  if (label_string.empty()) {
    throw std::invalid_argument("The provided training data file is empty.");
  }

  // The label will be the first (and only) character in label_string
  char label = label_string.at(0);
  stringstream image_lines;
  image_lines << label << kFileLineDelimiter;
  
  string first_line;
  getline(input, first_line);
  image_lines << first_line << kFileLineDelimiter;

  string line;
  getline(input, line);
  std::streampos original_position;

  // Add lines to the image until we read the next label or reach end of file
  while (!input.eof() && line.size() == first_line.size()) {
    // Adapted from https://stackoverflow.com/a/27331411
    original_position = input.tellg();
    image_lines << line << kFileLineDelimiter;
    getline(input, line);
  }

  // roll back the stream to stored position so we can re-access the next label
  if (!input.eof()) {
    input.seekg(original_position);
  } else {
    image_lines << line << kFileLineDelimiter; 
    // if the dataset is 1 image, add the last line
  }

  Image image;
  image_lines >> image;
  return image;
}

} // namespace naivebayes
