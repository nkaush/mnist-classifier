//
// Created by Neil Kaushikkar on 4/4/21.
//

#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace naivebayes {

/**
 * Contains enum encodings of all possible Shading types the model supports.
 */
enum class Shading{
  kWhite = 0,
  kBlack = 1,
  kGray = 2
};

/**
 * This abstraction represents an image with a 2D-vector of pixels and a label 
 * of the image contents and has functionality to describe the image shape. 
 */
class Image {
  public:
    // Stores all of the shading encodings since C++ does not support reflection
    static const std::vector<Shading> kDistinctShadingEncodings;

    // Stores rules about how to map characters in a file to Shading encodings
    static const std::map<char, Shading> kPixelShadings;
    
    Image() = default;
  
    /**
     * Instantiates an Image with the provided pixels and label.
     * @param pixels - a 2D-vector of the pixels to represent this image with
     * @param label - a char that indicates the label this image represents
     */
    Image(const std::vector<std::vector<Shading>>& pixels, char label);

    friend std::istream &operator>>(std::istream &input, Image& image);
  
    /**
     * Getter for the height of the image (not making assumptions about squares)
     * @return a size_t indicating the height of the image
     */
    size_t GetHeight() const;

    /**
     * Getter for the width of the image (not making assumptions about squares)
     * @return a size_t indicating the width of the image
     */
    size_t GetWidth() const;
    
    /**
     * Getter for the label corresponding to this image.
     * @return a char indicating the label of the image
     */
    char GetLabel() const;
    
    /**
     * Getter for the Shading at a particular pixel.
     * @param row - the index of the row of the pixel requested
     * @param column - the index of the column of the pixel requested
     * @return a Shading enum encoding of the requested pixel
     */
    Shading GetPixel(size_t row, size_t column) const;
  
    /**
     * Maps a digit encoding of a Shading enum to the Shading enum itself. This
     * function is required to deserialize a model from json since C++ does not
     * support reflection.
     * @param to_map - a string containing a digit the enum maps to
     * @return a Shading enum that corresponds to the given string
     * @throws std::invalid_argument if the string does not map to any enum
     */
    static Shading MapStringDigitEncodingToShading(const std::string& to_map);
  private:
    std::vector<std::vector<Shading>> pixels_;
    
    char label_;
};

}

#endif  // NAIVE_BAYES_IMAGE_H
