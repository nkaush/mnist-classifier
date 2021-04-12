//
// Created by Neil Kaushikkar on 4/5/21.
//

#ifndef NAIVE_BAYES_EXECUTABLE_LOGIC_H
#define NAIVE_BAYES_EXECUTABLE_LOGIC_H

#include "core/model.h"

namespace naivebayes {

/**
 * This class defines logic of how to handle flags passed in the command line.
 */
class ExecutableLogic {
  public: 
    /**
     * Initialized the logic object and the Model object it operates on.
     * @param laplace_factor - the smoothing to use when training the model
     */
    ExecutableLogic(size_t laplace_factor);
    
    /**
     * Executes the logic and returns the exit status code depending on whether 
     * the logic successfully executed (0) or not (1). This logic will fail
     * if train_flag and load_flag are both not empty strings since the model
     * must be either loaded or trained, not both. If any flag passed is an 
     * empty string, that logic is skipped. 
     * @param train_flag - a string indicating the file path of the dataset 
     *                     to train the model on
     * @param load_flag - a string indicating the file path of the model file
     *                    to load into the model in memory
     * @param save_flag - a string indicating the file to save this model to
     * @param test_flag - a string indicating the file path of the dataset 
     *                     to test the model on
     * @param confusion_flag - a string indicating the file path to save the 
     *                         confusion matrix generated from testing to
     * @param is_test_multi_threaded - bool indicating whether to multi thread 
     *                                 the testing, if we are to test the model
     * @return a 0 or 1, depending on the result of executing the CLI logic
     */
    int Execute(const std::string& train_flag, const std::string& load_flag,
                const std::string& save_flag, const std::string& test_flag, 
                const std::string& confusion_flag, bool is_test_multi_threaded);
  private:
    Model model_;
    
    // The delimiter to use when generating the csv file
    static constexpr char kCsvElementDelimiter = ',';
    
    // Messages to print throughout executing the user's request
    static const std::string kModelAccuracyMessage;
    static const std::string kTestingModelMessage;

    static const std::string kTrainingModelMessage;
    
    static const std::string kLoadingModelMessage;
    static const std::string kLoadingConflictMessage;
    
    static const std::string kSavingModelMessage;
    static const std::string kSavingConfusionMatrixMessage;
    static const std::string kConfusionMatrixColumnLabel;
    
    static const std::string kFinishedMessage;
    static const std::string kFailedMessage;


    /**
     * Save the model to the specified file path. Creates a file, if the
     * file does not exist, otherwise, overwrites the file.
     * @param file_path - a string indicating the file to save the model to
     */
    void SaveModel(const std::string& file_path) const;
    
    /**
     * Loads model from the specified file. Does nothing if file does not exist.
     * @param model_path - a string indicating the path to load the model from
     */
    void LoadModel(const std::string& model_path);
    
    /**
     * Trains model with provided dataset. Does nothing if file does not exist.
     * @param dataset_path - a string indicating the path of the dataset to load
     */
    void TrainModel(const std::string& dataset_path);
    
    /**
     * Tests the model linearly or concurrently, depending on the request to 
     * multi-thread with the dataset provided. Saves the confusion matrix to the
     * path given, if the confusion matrix path is not empty.
     * @param dataset_path - a string indicating the file path of the dataset to
     *                       test the model on
     * @param is_test_multi_threaded - a bool indicating whether to use multiple 
     *                                 threads when testing the model
     * @param confusion_csv_path - a string indicating the file path to save the
     *                             confusion matrix to          
     */
    void TestModel(const std::string& dataset_path, 
                   bool is_test_multi_threaded, 
                   const std::string& confusion_csv_path) const;
    
    /**
     * Writes the confusion matrix provided to a CSV file.
     * @param save_path - a string indicating the file path to save to
     * @param matrix - a 2D-vector representing the confusion matrix
     */
    void SaveConfusionMatrix(const std::string& save_path, 
                             const std::vector<std::vector<size_t>>& matrix) 
        const;
};

}

#endif  // NAIVE_BAYES_EXECUTABLE_LOGIC_H
