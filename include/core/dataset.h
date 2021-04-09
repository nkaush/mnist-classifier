//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_DATASET_H
#define NAIVE_BAYES_DATASET_H

#include <iostream>
#include <map>

#include "image.h"

namespace naivebayes {

/**
 * This class stores Images grouped by common class labels.
 */
class Dataset {
 public:
    /**
     * Initializes an empty Dataset object.
     */
    Dataset();

    /**
     * Gets all the images that belong to a single class.
     * @param class_label - the class to get images from
     * @return a vector of Images belonging to the specified class
     */
    const std::vector<Image>& GetImageGroup(char class_label) const;

    /**
     * Getter for the size of the dataset.
     * @return a size_t indicating the number of entries in this dataset
     */
    size_t GetSize() const;

    /**
     * Returns a vector of all the class labels from the dataset of Images.
     * @return a vector of char containing all the distinct class labels
     */
    std::vector<char> GetDistinctLabels() const;
    
    /**
     * Overloaded extraction operator - populates this Dataset with encoded 
     * representations of the images and labels in the input stream. 
     * @param input - an istream to read images from
     * @param dataset - the Dataset object to populate with images
     * @return the istream after the images have been extracted from it
     * @throws std::invalid_argument if any image is missing a label or if any
     * image is not the same uniform size as all other images
     */
    friend std::istream &operator>>(std::istream &input, Dataset& dataset);
  
  private:
    // Stores size of dataset so we don't have to compute size from a map
    size_t size_;
    
    // Groups Images by common labels
    std::map<char, std::vector<Image>> class_groups_;
    
    static constexpr char kFileLineDelimiter = '\n';
    
    /**
     * Parses the first image in an input stream to infer the dimension of all 
     * following images in the stream, as assumed in the project description
     * @param input - an istream to read the first image from
     * @return an Image struct representing an encoded form of the 1st image in 
     * the stream and is used to infer the dimension of all other images
     * @throws std::invalid_argument if the first image is missing a label
     */
    Image ParseFirstImage(std::istream& input) const;
};

} // namespace naivebayes

#endif  // NAIVE_BAYES_DATASET_H
