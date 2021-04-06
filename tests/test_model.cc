//
// Created by Neil Kaushikkar on 4/4/21.
//

#include <catch2/catch.hpp>

#include <core/model.h>

#include <fstream>
#include <sstream>

using naivebayes::Dataset;
using naivebayes::Shading;
using naivebayes::Model;
using std::stringstream;
using nlohmann::json;
using std::ifstream;
using std::vector;

TEST_CASE("Test Class Occurrence Likelihoods") {
  Model model = Model();
  Dataset dataset = Dataset();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
      "naive-bayes-nkaush/data/testing_dataset.txt";
  ifstream input(file_path);

  input >> dataset;
  model.Train(dataset);
  
  SECTION("Test likelihood of class label being 0") {
    REQUIRE(model.GetClassLikelihood('0') == Approx(6. / 11.));
  }
  
  SECTION("Test likelihood of class label being 1") {
    REQUIRE(model.GetClassLikelihood('1') == Approx(5. / 11.));
  }
}

TEST_CASE("Test Conditional Likelihoods") {
  Model model = Model();
  Dataset dataset = Dataset();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
      "naive-bayes-nkaush/data/testing_dataset.txt";
  ifstream input(file_path);
  
  input >> dataset;
  model.Train(dataset);
  
  SECTION("Test likelihood of being NOT shaded given class is 0") {
    /* Reference Counts: 
     * 3, 2, 2, 4
     * 2, 2, 1, 3
     * 2, 2, 2, 3
     * 3, 1, 1, 3
     */
    vector<vector<float>> correct_likelihoods = 
        { {0.5714, 0.4286, 0.4286, 0.7143},
          {0.4286, 0.4286, 0.2857, 0.5714},
          {0.4286, 0.4286, 0.4286, 0.5714},
          {0.5714, 0.2857, 0.2857, 0.5714} };
    
    for (size_t row = 0; row < 4; row++) {
      for (size_t column = 0; column < 4; column++) {
        REQUIRE(Approx(correct_likelihoods.at(row).at(column)).epsilon(0.01) == 
                model.GetFeatureLikelihood('0', Shading::kWhite, row, column));
      }
    }
  }

  SECTION("Test likelihood of being shaded given class is 0") {
    /* Reference Counts: 
     * 2, 3, 3, 1
     * 3, 3, 4, 2
     * 3, 3, 3, 2
     * 2, 4, 4, 2
     */
    vector<vector<float>> correct_likelihoods = 
        { {0.4286, 0.5714, 0.5714, 0.2857},
          {0.5714, 0.5714, 0.7143, 0.4286},
          {0.5714, 0.5714, 0.5714, 0.4286},
          {0.4286, 0.7143, 0.7143, 0.4286} };
    
    for (size_t row = 0; row < 4; row++) {
      for (size_t column = 0; column < 4; column++) {
        REQUIRE(Approx(correct_likelihoods.at(row).at(column)).epsilon(0.01) == 
                model.GetFeatureLikelihood('0', Shading::kBlack, row, column));
      }
    }
  }

  SECTION("Test likelihood of being NOT shaded given class is 1") {
    /* Reference Counts: 
     * 2, 1, 3, 4
     * 4, 1, 3, 4
     * 4, 1, 3, 4
     * 3, 2, 2, 4
     */
    vector<vector<float>> correct_likelihoods =
        { {0.5, 0.3333, 0.6667, 0.8333},
          {0.8333, 0.3333, 0.6667, 0.8333},
          {0.8333, 0.3333, 0.6667, 0.8333},
          {0.6667, 0.5, 0.5, 0.8333} };

    for (size_t row = 0; row < 4; row++) {
      for (size_t column = 0; column < 4; column++) {
        REQUIRE(Approx(correct_likelihoods.at(row).at(column)).epsilon(0.01) ==
                model.GetFeatureLikelihood('1', Shading::kWhite, row, column));
      }
    }
  }

  SECTION("Test likelihood of being shaded given class is 1") {
    /* Reference Counts: 
     * 2, 3, 1, 0
     * 0, 3, 1, 0
     * 0, 3, 1, 0
     * 1, 2, 2, 0
     */
    vector<vector<float>> correct_likelihoods =
        { {0.5, 0.6667, 0.3333, 0.1666},
          {0.1666, 0.6667, 0.3333, 0.1666},
          {0.1666, 0.6667, 0.3333, 0.1666},
          {0.3333, 0.5, 0.5, 0.1666} };

    for (size_t row = 0; row < 4; row++) {
      for (size_t column = 0; column < 4; column++) {
        REQUIRE(Approx(correct_likelihoods.at(row).at(column)).epsilon(0.01) ==
                model.GetFeatureLikelihood('1', Shading::kBlack, row, column));
      }
    }
  }
}

TEST_CASE("Test Model Serialization") {
  SECTION("Test empty model") {
    Model model = Model();
    stringstream serialized;
    
    serialized << model;
    
    REQUIRE(serialized.str() == "[]\n");
  }
  
  SECTION("Test model trained on testing dataset") {
    Model model = Model();
    Dataset dataset = Dataset();
    
    // Create the actual serialized model
    std::string dataset_path = "/Users/neilkaushikkar/Cinder/my-projects/"
        "naive-bayes-nkaush/data/testing_dataset.txt";
    ifstream dataset_input(dataset_path);

    dataset_input >> dataset;
    model.Train(dataset);

    stringstream model_as_string;
    model_as_string << model;
    json actual_serialized_model;
    model_as_string >> actual_serialized_model;
    
    // Load and parse the expected model
    std::string model_path = "/Users/neilkaushikkar/Cinder/my-projects/"
        "naive-bayes-nkaush/data/testing_model.json";
    ifstream model_input(model_path);
    
    // Adapted from https://stackoverflow.com/a/2602060
    std::string expected_serialized_string(
        (std::istreambuf_iterator<char>(model_input)),
        std::istreambuf_iterator<char>());
    json expected_serialized_model = json::parse(expected_serialized_string);

    REQUIRE(actual_serialized_model == expected_serialized_model);
  }
}

TEST_CASE("Test Model Deserialization") {
  std::string saved_model =
      "[\n"
      "  {\n"
      "    \"class_likelihood\": 0.0819,\n"
      "    \"label\": \"0\",\n"
      "    \"shading_likelihoods\": {\n"
      "      \"0\": [\n"
      "        [0.125, 0.241],\n"
      "        [0.101, 0.196]\n"
      "      ],\n"
      "      \"1\": [\n"
      "        [0.126, 0.173],\n"
      "        [0.415, 0.255]\n"
      "      ]\n"
      "    }\n"
      "  }\n"
      "]";

  stringstream serialized_model(saved_model);

  Model model = Model();
  serialized_model >> model;
  
  SECTION("Test class occurrence likelihood") {
    REQUIRE(model.GetClassLikelihood('0') == Approx(0.0819).epsilon(0.01));
  }

  SECTION("Test likelihood of being NOT shaded given class is 0") {
    vector<vector<float>> expected = { {0.125, 0.241}, {0.101, 0.196} };
    
    for (size_t row = 0; row < 2; row++) {
      for (size_t column = 0; column < 2; column++) {
        REQUIRE(Approx(expected.at(row).at(column)).epsilon(0.01) ==
                model.GetFeatureLikelihood('0', Shading::kWhite, row, column));
      }
    }
  }
  
  SECTION("Test likelihood of being shaded given class is 0") {
    vector<vector<float>> expected = { {0.126, 0.173}, {0.415, 0.255} };

    for (size_t row = 0; row < 2; row++) {
      for (size_t column = 0; column < 2; column++) {
        REQUIRE(Approx(expected.at(row).at(column)).epsilon(0.01) ==
                model.GetFeatureLikelihood('0', Shading::kBlack, row, column));
      }
    }
  }
}
