//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_DATASET_H
#define NAIVE_BAYES_DATASET_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "core/image.h"

namespace naivebayes {

class Dataset {
  public:
    Dataset();

    const std::vector<Image>& GetImageGroup(char group_class) const;

    size_t GetSize() const;

    std::vector<char> GetDistinctLabels() const;

    friend std::istream &operator>>(std::istream &input, Dataset& dataset);

  private:
    size_t size_;
    std::map<char, std::vector<Image>> class_groups_;

    static const std::map<char, size_t> kPixelShadings;

    void ValidateFilePath(const std::string& file_path) const;

    void AddImage(char label, const std::vector<std::string>& image_lines);

    Image ParseFirstImage(std::istream& in) const;

    std::vector<std::vector<size_t>> EncodeShadingStrings(
        const std::vector<std::string>& shading_strings) const;
};


} // namespace naivebayes

#endif  // NAIVE_BAYES_DATASET_H
