//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_DATASET_H
#define NAIVE_BAYES_DATASET_H

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <map>

namespace naivebayes {

enum class Shading {
  kWhite,
  kBlack
};

struct Image {
  std::vector<std::vector<Shading>> pixels_;
  char label_;

  Image(std::vector<std::vector<Shading>> pixels, char label) :
    pixels_(std::move(pixels)), label_(label) {}

  size_t GetHeight() const;
  size_t GetWidth() const;
};

class Dataset {
  public:
    static const std::vector<Shading> kDistinctShadingEncodings;

    Dataset();

    const std::vector<Image>& GetImageGroup(char class_label) const;

    size_t GetSize() const;

    std::vector<char> GetDistinctLabels() const;

    friend std::istream &operator>>(std::istream &input, Dataset& dataset);
  private:
    size_t size_;
    std::map<char, std::vector<Image>> class_groups_;

    static const std::map<char, Shading> kPixelShadings;

    void ValidateFilePath(const std::string& file_path) const;

    void AddImage(char label, const std::vector<std::string>& image_lines);

    Image ParseFirstImage(std::istream& in) const;

    std::vector<std::vector<Shading>> EncodeShadingStrings(
        const std::vector<std::string>& shading_strings) const;
};

} // namespace naivebayes

#endif  // NAIVE_BAYES_DATASET_H
