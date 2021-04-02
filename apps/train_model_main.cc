#include <core/dataset.h>
#include <core/model.h>

#include <fstream>
#include <iostream>

using naivebayes::Dataset;
using naivebayes::Image;
using naivebayes::Model;

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.

  std::string path = "/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/trainingimagesandlabels.txt";
  std::string mock = "/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/mock_data.txt";
  Dataset dataset = Dataset();

  // Adapted from: http://www.cplusplus.com/reference/istream/istream/istream/
  std::filebuf fb;
  if (fb.open(path, std::ios::in)) {
    std::istream input(&fb);

    input >> dataset;
  }

  Model model = Model();
  model.Train(dataset);

  return 0;
}
