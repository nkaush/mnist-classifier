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
DEFINE_bool(verbose, false, "Whether to print the current index when testing.");

using naivebayes::ExecutableLogic;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  // Prevent a user from setting the unsigned smoothing factor to 0
  if (FLAGS_smoothing <= 0) {
    FLAGS_smoothing = naivebayes::Model::kDefaultLaplaceSmoothingFactor;
  }
  
  ExecutableLogic logic = ExecutableLogic(FLAGS_smoothing);
  
  return logic.Execute(FLAGS_train, FLAGS_load, FLAGS_save, FLAGS_test, 
                       FLAGS_confusion, FLAGS_verbose);
}
