//
// Created by Neil Kaushikkar on 4/5/21.
//

#ifndef NAIVE_BAYES_EXECUTABLE_LOGIC_H
#define NAIVE_BAYES_EXECUTABLE_LOGIC_H

#include "core/model.h"

namespace naivebayes {

class ExecutableLogic {
  public: 
    ExecutableLogic();
    
    int Execute(const std::string& train_flag, const std::string& load_flag,
                const std::string& save_flag);
  private:
    Model model_;
    
    void SaveModel(const std::string& file_path) const;
    
    void LoadModel(const std::string& model_path);
    
    void TrainModel(const std::string& dataset_path);
};

}

#endif  // NAIVE_BAYES_EXECUTABLE_LOGIC_H
