//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <iostream>
#include <fstream>

#include "core/dataset.h"

namespace naivebayes {

using std::istream;
using std::vector;
using std::string;
using std::map;

const map<char, Shading> Dataset::kPixelShadings =
    {{' ', Shading::kWhite}, {'+', Shading::kBlack}, {'#', Shading::kBlack}};

const vector<Shading> Dataset::kDistinctShadingEncodings =
    {Shading::kWhite, Shading::kBlack};

size_t Image::GetHeight() const {
  return pixels_.size();
}

size_t Image::GetWidth() const {
  return pixels_.at(0).size();
}

Dataset::Dataset() : size_(0) {}

// TODO add somewhere
void Dataset::ValidateFilePath(const string& file_path) const {
  std::ifstream located_file(file_path);

  if (!located_file.is_open()) {
    throw std::invalid_argument("The file path is not valid.");
  }

  located_file.close();
}

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
  // TODO check for images out of shape
  Image first_image = dataset.ParseFirstImage(input);

  size_t image_height = first_image.pixels_.size();
  char first_image_label = first_image.label_;

  // Get the first image so we know dimensions to search for other images
  std::pair<char, vector<Image>> first_class(first_image_label, {first_image});
  dataset.class_groups_.insert(first_class);
  dataset.size_++;

  string current_label;
  while (getline(input, current_label)) { // continue while there is a label
    string next_line;
    vector<string> lines;
    // aggregate all the lines in the image, assuming all images are same size
    for (size_t line_index = 0; line_index < image_height; line_index++) {
      getline(input, next_line);
      lines.push_back(next_line);
    }

    char label = current_label.at(0);  // assume, for now the label is 1 char
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

Image Dataset::ParseFirstImage(istream& in) const {
  string label_string;
  getline(in, label_string);
  char label = label_string.at(0);

  string first_line;
  getline(in, first_line);
  size_t image_width = first_line.size();

  vector<string> lines = {first_line};

  string line;
  getline(in, line);

  std::streampos original_position;

  // Add lines to the image until we read the next label
  while (line.size() == image_width) {
    // Adapted from https://stackoverflow.com/a/27331411
    original_position = in.tellg();
    lines.push_back(line);
    getline(in, line);
  }

  // roll back the stream to stored position so we can re-access the next label
  in.seekg(original_position);

  return Image(EncodeShadingStrings(lines), label);
}

vector<vector<Shading>> Dataset::EncodeShadingStrings(
    const vector<string>& shading) const {
  vector<vector<Shading>> pixel_grid;

  for (const string& row : shading) {
    vector<Shading> pixel_row;

    for (char pixel : row) {
      pixel_row.push_back(kPixelShadings.at(pixel));
    }

    pixel_grid.push_back(pixel_row);
  }

  return pixel_grid;
}

Shading Dataset::MapStringToShading(std::string to_map) {
  for (const Shading& shading : kDistinctShadingEncodings) {
    string encoding_string = std::to_string(static_cast<int>(shading));
    if (to_map == encoding_string) {
      return shading;
    }
  }
  
  throw std::invalid_argument("The shading encoding string provided is invalid.");
}

} // namespace naivebayes
