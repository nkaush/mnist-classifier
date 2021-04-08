#include <gflags/gflags.h>

#include <core/executable_logic.h>

// Define the command line arguments and set the default value to empty strings
DEFINE_string(train, "", "The file path to the dataset to train the model on.");
DEFINE_string(save, "", "The file path to save the trained model to.");
DEFINE_string(load, "", "The file path to load a pre-trained model from.");
DEFINE_string(test, "", "The file path to the dataset to test the model on.");
DEFINE_string(confusion, "", "The file path to save the confusion matrix to.");
DEFINE_uint32(smoothing, naivebayes::Model::kDefaultLaplaceSmoothingFactor,
              "The Laplace smoothing factor to use in calculating likelihoods.");
DEFINE_bool(multithread, false, "Whether to multi-thread the tests when "
            "validating the model");

using naivebayes::ExecutableLogic;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  ExecutableLogic logic = ExecutableLogic();
  
  return logic.Execute(FLAGS_train, FLAGS_load, FLAGS_save, FLAGS_test, 
                       FLAGS_confusion, FLAGS_multithread);
}
