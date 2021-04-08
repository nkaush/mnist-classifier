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
     * Default constructor that initializes and empty model that will be filled 
     * by files passed in from the command line or trained with files passed
     * in via the command line.
     */
    ExecutableLogic();
    
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
    
    static constexpr char kCsvElementDelimiter = ',';
    static constexpr char kCsvLineDelimiter = '\n';
    
    static void ValidateFilePath(const std::string& file_path) ;
    
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
    
    void TestModel(const std::string& dataset_path, 
                   bool is_test_multi_threaded, 
                   const std::string& confusion_csv_path) const;
    
    void SaveConfusionMatrix(const std::string& save_path, 
                             const std::vector<std::vector<size_t>>& matrix) 
        const;
};

}

#endif  // NAIVE_BAYES_EXECUTABLE_LOGIC_H
