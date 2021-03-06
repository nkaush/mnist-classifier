//
// Created by Neil Kaushikkar on 4/1/21.
//

#include <catch2/catch.hpp>

#include <core/dataset.h>

#include <fstream>
#include <sstream>

using naivebayes::Dataset;
using naivebayes::Shading;
using naivebayes::Image;
using std::stringstream;
using std::fstream;
using std::vector;
using std::string;

TEST_CASE("Test Reading Mock Dataset") {
  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
      "naive-bayes-nkaush/data/testing_train_dataset_4x4.txt";
  fstream input(file_path);
  Dataset dataset;

  input >> dataset;
  
  SECTION("Test dataset size") {
    REQUIRE(dataset.GetSize() == 9);
  }
  
  SECTION("Test dataset inferring correct labels") {
    REQUIRE(dataset.GetDistinctLabels() == vector<char>({'0', '1'}));
  }
  
  SECTION("Test dataset correctly encodes images with label '0'") {
    // Use a shorthand to make readability easier and shorter
    Shading b = Shading::kBlack;
    Shading w = Shading::kWhite;
    
    vector<vector<vector<Shading>>> images;
    
    images.push_back({{b, b, b, w}, {b, w, b, w}, {b, b, b, w}, {w, w, w, w}});
    images.push_back({{w, w, w, w}, {b, b, b, w}, {b, w, b, w}, {b, b, b, w}});
    images.push_back({{w, w, w, w}, {w, b, b, b}, {w, b, w, b}, {w, b, b, b}});
    images.push_back({{b, b, b, w}, {b, w, b, w}, {b, w, b, w}, {b, b, b, w}});
    images.push_back({{w, b, b, b}, {w, b, w, b}, {w, b, w, b}, {w, b, b, b}});
    
    vector<Image> zero_class_images = dataset.GetImageGroup('0');
    for (size_t i = 0; i < zero_class_images.size(); i++) {
      for (size_t row = 0; row < zero_class_images.at(i).GetHeight(); row++) {
        for (size_t col = 0; col < zero_class_images.at(i).GetWidth(); col++) {
          REQUIRE(zero_class_images.at(i).GetPixel(row, col) == 
                  images.at(i).at(row).at(col));
        }
      }
    }
  }

  SECTION("Test dataset correctly encodes images with label '1'") {
    // Use a shorthand to make readability easier
    Shading b = Shading::kBlack;
    Shading w = Shading::kWhite;

    vector<vector<vector<Shading>>> images;

    images.push_back({{w, w, b, w}, {w, w, b, w}, {w, w, b, w}, {w, w, b, w}});
    images.push_back({{w, b, w, w}, {w, b, w, w}, {w, b, w, w}, {w, b, w, w}});
    images.push_back({{b, b, w, w}, {w, b, w, w}, {w, b, w, w}, {w, w, w, w}});
    images.push_back({{b, b, w, w}, {w, b, w, w}, {w, b, w, w}, {b, b, b, w}});

    vector<Image> one_class_images = dataset.GetImageGroup('1');
    for (size_t i = 0; i < one_class_images.size(); i++) {
      for (size_t row = 0; row < one_class_images.at(i).GetHeight(); row++) {
        for (size_t col = 0; col < one_class_images.at(i).GetWidth(); col++) {
          REQUIRE(one_class_images.at(i).GetPixel(row, col) ==
                  images.at(i).at(row).at(col));
        }
      }
    }
  }
}

TEST_CASE("Test Single Image Entry") {
  string image_text = 
      "0\n"
      "### \n"
      "# # \n"
      "### \n"
      "    ";
  stringstream input(image_text);
  Dataset dataset = Dataset();
  
  input >> dataset;
  
  SECTION("Test dataset correctly encodes image") {
    Shading b = Shading::kBlack;
    Shading w = Shading::kWhite;

    vector<vector<Shading>> correct_grid = {{b, b, b, w},
                                            {b, w, b, w},
                                            {b, b, b, w},
                                            {w, w, w, w}};
    
    const Image& to_check = dataset.GetImageGroup('0').at(0);
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == to_check.GetPixel(row, col));
      }
    }
  }
  
  SECTION("Test dataset accurately records size") {
    REQUIRE(dataset.GetSize() == 1);
  }

  SECTION("Test dataset inferring correct labels") {
    REQUIRE(dataset.GetDistinctLabels() == vector<char>({'0'}));
  }
}

TEST_CASE("Test Loading Images of Different Size - 5x5") {
  string image_text =
      "0\n"
      "###  \n"
      "# #  \n"
      "###  \n"
      "     \n"
      "     \n"
      "1\n"
      " ##  \n"
      "  #  \n"
      "  #  \n"
      "  #  \n"
      " ### \n";
  stringstream input(image_text);
  Dataset dataset = Dataset();

  input >> dataset;

  Shading b = Shading::kBlack;
  Shading w = Shading::kWhite;
  
  SECTION("Test dataset correctly encodes images with label '0'") {
    vector<vector<Shading>> correct_grid = {{b, b, b, w, w},
                                            {b, w, b, w, w},
                                            {b, b, b, w, w},
                                            {w, w, w, w, w},
                                            {w, w, w, w, w}};
    
    const Image& to_check = dataset.GetImageGroup('0').at(0);
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == to_check.GetPixel(row, col));
      }
    }
  }

  SECTION("Test dataset correctly encodes images with label '1'") {
    vector<vector<Shading>> correct_grid = {{w, b, b, w, w},
                                            {w, w, b, w, w},
                                            {w, w, b, w, w},
                                            {w, w, b, w, w},
                                            {w, b, b, b, w}};
    
    const Image& to_check = dataset.GetImageGroup('1').at(0);
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == to_check.GetPixel(row, col));
      }
    }
  }

  SECTION("Test dataset accurately records size") {
    REQUIRE(dataset.GetSize() == 2);
  }

  SECTION("Test dataset inferring correct labels") {
    REQUIRE(dataset.GetDistinctLabels() == vector<char>({'0', '1'}));
  }
}

TEST_CASE("Test Loading Images of Different Size - 6x6") {
  string image_text =
      "0\n"
      "      \n"
      " #### \n"
      " #  # \n"
      " #  # \n"
      " #### \n"
      "      \n"
      "1\n"
      "      \n"
      "  ##  \n"
      "   #  \n"
      "   #  \n"
      "  ### \n"
      "      \n";
  stringstream input(image_text);
  Dataset dataset = Dataset();

  input >> dataset;

  Shading b = Shading::kBlack;
  Shading w = Shading::kWhite;

  SECTION("Test dataset correctly encodes images with label '0'") {
    vector<vector<Shading>> correct_grid = {{w, w, w, w, w, w},
                                            {w, b, b, b, b, w},
                                            {w, b, w, w, b, w},
                                            {w, b, w, w, b, w},
                                            {w, b, b, b, b, w},
                                            {w, w, w, w, w, w}};
    
    const Image& to_check = dataset.GetImageGroup('0').at(0);
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == to_check.GetPixel(row, col));
      }
    }
  }

  SECTION("Test dataset correctly encodes images with label '1'") {
    vector<vector<Shading>> correct_grid = {{w, w, w, w, w, w},
                                            {w, w, b, b, w, w},
                                            {w, w, w, b, w, w},
                                            {w, w, w, b, w, w},
                                            {w, w, b, b, b, w},
                                            {w, w, w, w, w, w}};
    
    const Image& to_check = dataset.GetImageGroup('1').at(0);
    for (size_t row = 0; row < correct_grid.size(); row++) {
      for (size_t col = 0; col < correct_grid.at(row).size(); col++) {
        REQUIRE(correct_grid.at(row).at(col) == to_check.GetPixel(row, col));
      }
    }
  }

  SECTION("Test dataset accurately records size") {
    REQUIRE(dataset.GetSize() == 2);
  }

  SECTION("Test dataset inferring correct labels") {
    REQUIRE(dataset.GetDistinctLabels() == vector<char>({'0', '1'}));
  }
}

TEST_CASE("Test invalid dataset files") {
  SECTION("Test empty file") {
    stringstream input("");
    Dataset dataset = Dataset();

    REQUIRE_THROWS_AS(input >> dataset, std::invalid_argument);
  }
  
  SECTION("Test image missing label") {
    string image_text = 
        "\n"
        "#  \n"
        "#  \n"
        "#  \n";
    stringstream input(image_text);
    Dataset dataset = Dataset();

    REQUIRE_THROWS_AS(input >> dataset, std::invalid_argument);
  }

  SECTION("Test second image not uniform width") {
    string image_text =
        "1\n"
        "#  \n"
        "#  \n"
        "#  \n"
        "1\n"
        " # \n"
        " #       \n"
        " # \n";
    stringstream input(image_text);
    Dataset dataset = Dataset();

    REQUIRE_THROWS_AS(input >> dataset, std::invalid_argument);
  }

  SECTION("Test second image not uniform height") {
    string image_text =
        "1\n"
        "#  \n"
        "#  \n"
        "#  \n"
        "1\n"
        " # \n"
        " # \n";
    stringstream input(image_text);
    Dataset dataset = Dataset();

    REQUIRE_THROWS_AS(input >> dataset, std::invalid_argument);
  }
}