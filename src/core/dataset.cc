//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <iostream>

#include "core/dataset.h"
#include "core/image.h"

namespace naivebayes {

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
  char first_image_label = first_image.GetLabel();

  // Get the first image so we know dimensions to search for other images
  std::pair<char, vector<Image>> first_class(first_image_label, {first_image});
  dataset.class_groups_.insert(first_class);
  dataset.size_++;

  string current_label;
  while (!input.eof() && getline(input, current_label)) { 
    // continue while there is a label
    string next_line;
    vector<string> lines;
    // aggregate all the lines in the image, assuming all images are same size
    for (size_t line_index = 0; line_index < image_height; line_index++) {
      getline(input, next_line);
      
      // Only need to check width because if height was off line would be a label
      if (next_line.size() != first_image.GetWidth()) {
        // label lengths are also different than line lengths
        throw std::invalid_argument("The images are not of uniform size");
      }
      lines.push_back(next_line);
    }
    
    if (current_label.empty()) {
      throw std::invalid_argument("Image is missing a label.");
    }
    char label = current_label.at(0);  // assume, for now the label is 1st char
    dataset.AddImage(label, lines);
  }

  return input;
}

void Dataset::AddImage(char label, const vector<string>& image_lines) {
  vector<vector<Shading>> encoding = EncodeShadingStrings(image_lines);
  auto group_iterator = class_groups_.find(label);

  // If we have not seen an image from this class, add its label
  if (group_iterator == class_groups_.end()) {
    // Insert the group into the dataset
    std::pair<char, vector<Image>> new_class(label, {});
    class_groups_.insert(new_class);
    // Re-find the class, so we can add to it
    group_iterator = class_groups_.find(label);
  }
  // Create a new Image here and add to corresponding class group
  Image image = Image(encoding, label);
  group_iterator->second.push_back(image);
  size_++;
}

Image Dataset::ParseFirstImage(istream& input) const {
  string label_string;
  getline(input, label_string);
  
  // If the file is empty, the first line will be empty!
  if (label_string.empty()) {
    throw std::invalid_argument("The provided training data file is empty.");
  }
  
  string first_line;
  getline(input, first_line);
  vector<string> lines = {first_line};

  string line;
  getline(input, line);
  std::streampos original_position;

  // Add lines to the image until we read the next label or reach end of file
  while (!input.eof() && line.size() == first_line.size()) {
    // Adapted from https://stackoverflow.com/a/27331411
    original_position = input.tellg();
    lines.push_back(line);
    getline(input, line);
  }

  // roll back the stream to stored position so we can re-access the next label
  if (!input.eof()) {
    input.seekg(original_position);
  } else {
    lines.push_back(line); // if the dataset is 1 image, add the last line
  }

  // The label will be the first (and only) character in label_string
  return Image(EncodeShadingStrings(lines), label_string.at(0));
}

vector<vector<Shading>> Dataset::EncodeShadingStrings(
    const vector<string>& shading) const {
  vector<vector<Shading>> pixel_grid;

  for (const string& row : shading) {
    vector<Shading> pixel_row;

    for (char pixel : row) {
      pixel_row.push_back(Image::kPixelShadings.at(pixel));
    }

    pixel_grid.push_back(pixel_row);
  }

  return pixel_grid;
}

} // namespace naivebayes
