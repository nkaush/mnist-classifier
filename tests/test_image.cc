//
// Created by Neil Kaushikkar on 4/6/21.
//

#include <catch2/catch.hpp>

#include <core/image.h>

using naivebayes::Shading;
using naivebayes::Image;

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
}
