//
// Created by Neil Kaushikkar on 4/14/21.
//

#include <catch2/catch.hpp>

#include <core/model.h>

#include <fstream>
#include <sstream>

using naivebayes::Dataset;
using naivebayes::Image;
using naivebayes::Model;
using std::stringstream;
using std::ifstream;
using std::string;
using std::vector;

using LongMatrix = vector<vector<size_t>>;

TEST_CASE("Test Likelihood Score Calculation and Image Classification on 4x4") {
  Model model = Model();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_model.json";
  ifstream model_input(file_path);
  model_input >> model;
  
  Dataset testing_dataset;
  std::string test_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_test_dataset_4x4.txt";
  ifstream dataset_input(test_path);
  dataset_input >> testing_dataset;

  vector<char> known_labels = testing_dataset.GetDistinctLabels();

  SECTION("Test calculation & classification on '0' image from train dataset") {
    Image image = testing_dataset.GetImageGroup('0').at(1);
    vector<float> expected_scores = {-4.264, -6.1574};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '0');
  }

  SECTION("Test calculation & classification on '1' image from train dataset") {
    Image image = testing_dataset.GetImageGroup('1').at(1);
    vector<float> expected_scores = {-5.0598, -2.9532};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '1');
  }

  SECTION("Test calculation & classification on '0' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('0').at(0);
    vector<float> expected_scores = {-5.3097, -6.0502};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '0');
  }

  SECTION("Test calculation & classification on '1' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('1').at(0);
    vector<float> expected_scores = {-5.0598, -4.8563};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '1');
  }
}

TEST_CASE("Test Likelihood Score Calculation and Image Classification on 5x5") {
  Model model = Model();
  Dataset train_dataset = Dataset();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_train_dataset_5x5.txt";
  ifstream train_input(file_path);
  train_input >> train_dataset;
  model.Train(train_dataset);

  Dataset testing_dataset;
  std::string test_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_test_dataset_5x5.txt";
  ifstream test_input(test_path);
  test_input >> testing_dataset;
  
  vector<char> known_labels = testing_dataset.GetDistinctLabels();

  SECTION("Test calculation & classification on '0' image from train dataset") {
    Image image = testing_dataset.GetImageGroup('0').at(0);
    vector<float> expected_scores = {-5.6045, -7.3288};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '0');
  }

  SECTION("Test calculation & classification on '1' image from train dataset") {
    Image image = testing_dataset.GetImageGroup('1').at(0);
    vector<float> expected_scores = {-6.5588, -4.2162};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '1');
  }

  SECTION("Test calculation & classification on '0' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('0').at(1);
    vector<float> expected_scores = {-6.5588, -6.9028};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '0');
  }

  SECTION("Test calculation & classification on '1' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('1').at(1);
    vector<float> expected_scores = {-6.0817, -4.8182};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '1');
  }

  SECTION("Test forcing mis-classification on '0' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('0').at(2);
    vector<float> expected_scores = {-7.9902, -7.6072};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '1');
  }

  SECTION("Test forcing mis-classification on '1' image NOT in train dataset") {
    Image image = testing_dataset.GetImageGroup('1').at(2);
    vector<float> expected_scores = {-7.0359, -7.7548};

    for (size_t idx = 0; idx < known_labels.size(); idx++) {
      REQUIRE(Approx(expected_scores.at(idx)).epsilon(0.01) ==
              model.CalculateLikelihoodScore(known_labels.at(idx), image));
    }

    REQUIRE(model.Classify(image) == '0');
  }
}

TEST_CASE("Test Model Testing on 4x4 Images") {
  Model model = Model();
  Dataset train_dataset = Dataset();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_train_dataset_4x4.txt";
  ifstream train_input(file_path);
  train_input >> train_dataset;
  model.Train(train_dataset);

  Dataset testing_dataset;
  std::string test_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_test_dataset_4x4.txt";
  ifstream test_input(test_path);
  test_input >> testing_dataset;
  
  SECTION("Test linear test returns correct confusion matrix on 4x4") {
    LongMatrix expected {{2, 0},
                         {0, 2}};
    LongMatrix actual = model.Test(testing_dataset, false);

    REQUIRE(actual == expected);
    REQUIRE(Model::CalculateAccuracy(actual) == Approx(1));
  }

  SECTION("Test multi-threaded test returns correct confusion matrix on 4x4") {
    LongMatrix expected {{2, 0},
                         {0, 2}};
    LongMatrix actual = model.Test(testing_dataset, false);

    REQUIRE(actual == expected);
    REQUIRE(Model::CalculateAccuracy(actual) == Approx(1));
  }
}

TEST_CASE("Test Model Testing on 5x5 Images") {
  Model model = Model();
  Dataset train_dataset = Dataset();

  // Need long verbose filepath since Cmake/Cinder can't locate local file path
  std::string file_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_train_dataset_5x5.txt";
  ifstream train_input(file_path);
  train_input >> train_dataset;
  model.Train(train_dataset);

  Dataset testing_dataset;
  std::string test_path = "/Users/neilkaushikkar/Cinder/my-projects/"
                          "naive-bayes-nkaush/data/testing_test_dataset_5x5.txt";
  ifstream test_input(test_path);
  test_input >> testing_dataset;
  
  SECTION("Test linear test returns correct confusion matrix on 4x4") {
    LongMatrix expected {{2, 1},
                         {1, 2}};
    LongMatrix actual = model.Test(testing_dataset, false);

    REQUIRE(actual == expected);
    REQUIRE(Model::CalculateAccuracy(actual) == Approx(0.66667));
  }

  SECTION("Test multi-threaded test returns correct confusion matrix on 4x4") {
    LongMatrix expected {{2, 1},
                         {1, 2}};
    LongMatrix actual = model.Test(testing_dataset, false);

    REQUIRE(actual == expected);
    REQUIRE(Model::CalculateAccuracy(actual) == Approx(0.66667));
  }
}
