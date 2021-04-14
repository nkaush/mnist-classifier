//
// Created by Neil Kaushikkar on 4/6/21.
//

#include <core/image.h>

#include <catch2/catch.hpp>
#include <sstream>

using naivebayes::Shading;
using naivebayes::Image;
using std::stringstream;
using std::string;
using std::vector;

TEST_CASE("Test Character to Shading Mapping") {
  SECTION("Test empty string") {
    REQUIRE_THROWS_AS(Image::MapStringDigitEncodingToShading(""), 
                      std::invalid_argument);
  }

  SECTION("Test single character string with unknown character") {
    REQUIRE_THROWS_AS(Image::MapStringDigitEncodingToShading("A"),
                      std::invalid_argument);
  }

  SECTION("Test multi character string") {
    REQUIRE_THROWS_AS(Image::MapStringDigitEncodingToShading("aBcDe"),
                      std::invalid_argument);
  }

  SECTION("Test white shading") {
    REQUIRE(Image::MapStringDigitEncodingToShading("0") == Shading::kWhite);
  }

  SECTION("Test black shading") {
    REQUIRE(Image::MapStringDigitEncodingToShading("1") == Shading::kBlack);
  }

  SECTION("Test gray shading") {
    REQUIRE(Image::MapStringDigitEncodingToShading("2") == Shading::kGray);
  }
}

TEST_CASE("Test Image Extraction Operator on 5x5 Image") {
  string image_text =
      "1\n"
      " ##  \n"
      "  +  \n"
      "  +  \n"
      "  +  \n"
      " ### ";
  stringstream pixel_data(image_text);

  Image image;
  pixel_data >> image;

  Shading b = Shading::kBlack;
  Shading w = Shading::kWhite;
  Shading g = Shading::kGray;

  SECTION("Test extraction operator reads label correctly") {
    REQUIRE(image.GetLabel() == '1');
  }

  SECTION("Test extraction operator reads all lines provided") {
    REQUIRE(image.GetHeight() == 5);
  }

  SECTION("Test extraction operator reads all characters in each line") {
    REQUIRE(image.GetWidth() == 5);
  }

  SECTION("Test extraction operator encodes chars correctly") {
    vector<vector<Shading>> correct_grid = {{w, b, b, w, w},
                                            {w, w, g, w, w},
                                            {w, w, g, w, w},
                                            {w, w, g, w, w},
                                            {w, b, b, b, w}};

    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == image.GetPixel(row, col));
      }
    }
  }
}

TEST_CASE("Test Image Extraction Operator on 6x6 Image") {
  string image_text =
      "0\n"
      "      \n"
      " +##+ \n"
      " #  # \n"
      " #  # \n"
      " +##+ \n"
      "      ";
  stringstream pixel_data(image_text);
  
  Image image;
  pixel_data >> image;

  Shading b = Shading::kBlack;
  Shading w = Shading::kWhite;
  Shading g = Shading::kGray;

  SECTION("Test extraction operator reads label correctly") {
    REQUIRE(image.GetLabel() == '0');
  }

  SECTION("Test extraction operator reads all lines provided") {
    REQUIRE(image.GetHeight() == 6);
  }

  SECTION("Test extraction operator reads all characters in each line") {
    REQUIRE(image.GetWidth() == 6);
  }

  SECTION("Test extraction operator encodes chars correctly") {
    vector<vector<Shading>> correct_grid = {{w, w, w, w, w, w},
                                            {w, g, b, b, g, w},
                                            {w, b, w, w, b, w},
                                            {w, b, w, w, b, w},
                                            {w, g, b, b, g, w},
                                            {w, w, w, w, w, w}};
    
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == image.GetPixel(row, col));
      }
    }
  }
}
