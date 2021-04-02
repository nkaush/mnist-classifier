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
  Dataset dataset = Dataset();

  // Adapted from: http://www.cplusplus.com/reference/istream/istream/istream/
  std::filebuf fb;
  if (fb.open (path, std::ios::in)) {
    std::istream in(&fb);

    in >> dataset;
  }

  Model model = Model();
  model.Train(dataset);
//  Image first = dataset.GetImage(0);
//
//  for (const std::vector<size_t>& row : first.GetPixels()) {
//    for (size_t pxl : row) {
//      std::cout << pxl;
//    }
//    std::cout << std::endl;
//  }
//
//  std::cout << std::endl;
//  Image next = dataset.GetImage(1);
//
//  for (const std::vector<size_t>& row : next.GetPixels()) {
//    for (size_t pxl : row) {
//      std::cout << pxl;
//    }
//    std::cout << std::endl;
//  }

  return 0;
}
