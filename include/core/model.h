//
// Created by Neil Kaushikkar on 4/1/21.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <nlohmann/json.hpp>
#include <thread>
#include <future>

#include "core/dataset.h"

namespace naivebayes {

/**
 * This struct serves as an abstraction to hold the likelihood of specific 
 * features occurring for the class that this struct represents
 */
struct Classification {
  float class_likelihood_;
  std::map<Shading, std::vector<std::vector<float>>> shading_likelihoods_;
};

/**
 * Stores the conditional likelihoods for all features and the likelihoods of 
 * all classes based on the dataset provided when training this model
 */
class Model {
  public:
    // The smoothing factor to use when calculating feature likelihoods
    static constexpr size_t kDefaultLaplaceSmoothingFactor = 1;
    
    /**
     * Default constructor. The model must be initialized by either training
     * it with a Dataset or by loading a saved model from a stream.
     */
    Model(float laplace_smoothing=kDefaultLaplaceSmoothingFactor);

    /**
     * Trains the model with the provided Dataset. Initializes the model.
     * @param dataset - a Dataset object containing encoded images from a stream 
     */
    void Train(const Dataset& dataset);
    
    /**
     * Tests the model by classifying each image in the dataset sequentially 
     * and generating a confusion matrix displaying the count of predicted 
     * labels with respect to the count of actual labels.
     * @param dataset - a Dataset object containing Images & their actual labels
     * @return 2D-vector representing a confusion matrix generated from testing
     */
    std::vector<std::vector<size_t>> Test(const Dataset& dataset) const;

    /**
     * Tests the model by classifying each image in the dataset concurrently by 
     * spawning multiple threads and generating a confusion matrix displaying 
     * the count of predicted labels with respect to the count of actual labels.
     * @param dataset - a Dataset object containing Images & their actual labels
     * @return 2D-vector representing a confusion matrix generated from testing
     */
    std::vector<std::vector<size_t>> MultiThreadedTest(
        const Dataset& dataset) const;
    
    /**
     * Classify the given Image by comparing its features to the model.
     * @param image - an Image object to classify
     * @return a char indicating the predicted label of the Image
     */
    char Classify(const Image& image) const;
    
    /**
     * Getter for the likelihood of the occurrence of a class.
     * @param class_label - the label to retrieve the likelihood for
     * @return a float indicating the likelihood of the occurrence of a class
     */
    float GetClassLikelihood(char class_label) const;
    
    /**
     * Getter for the likelihood of a particular feature at the provided row and
     * column appearing with the specified Shading for some specified class.
     * @param class_label - the label to retrieve the likelihood for
     * @param shading - a Shading enum corresponding to a likelihood to retrieve
     * @param row - size_t indicating the y-axis index of a feature to retrieve
     * @param column - size_t indicating the x-axis index of feature to retrieve
     * @return a float indicating the likelihood of the specified feature
     */
    float GetFeatureLikelihood(char class_label, Shading shading, 
                               size_t row, size_t column) const;
    
    std::map<char, size_t> GetLabelIndices() const;
    
    /**
     * Calculates the accuracy of the model by tallying the number of correct
     * predictions in a confusion matrix and dividing by the total predictions.
     * @param confusion_matrix - a 2D vector indicating the confusion matrix
     *                           generated by testing the model
     * @return a float between 0 and 1 indicating the accuracy of the model
     */
    static float CalculateAccuracy(
        const std::vector<std::vector<size_t>>& confusion_matrix);
    
    /**
     * Overloaded insertion operator - streams model into a given output stream.
     * @param output - an ostream to insert the model into
     * @param model - a Model object to insert into the ostream
     * @return the ostream after the model has been inserted
     */
    friend std::ostream &operator<<(std::ostream& output, const Model& model);

    /**
     * Overloaded extraction operator - extracts a model from the given stream.
     * @param input - an istream to extract a serialized model from
     * @param model - a Model object to fill with the serialized model 
     * @return the istream after the model has been extracted from it
     */
    friend std::istream &operator>>(std::istream& input, Model& model);
    
  private:
    // Stores the likelihood of each class and all features for each class
    std::map<char, Classification> classifications_;
    
    float laplace_smoothing_;
    
    // The spacing schema to use when generating the serialized model
    static constexpr size_t kJsonSchemaSpacing = 2;
    
    // Keys that define the structure of the JSON schema
    static const std::string kJsonSchemaLabelKey; 
    static const std::string kJsonSchemaClassKey; 
    static const std::string kJsonSchemaShadingKey; 
    
    /**
     * Calculates the conditional likelihood of a Shading appearing in an image
     * for all pixels in a provided vector of images with the same label.
     * @param class_group - a vector if Images with the same label
     * @param label_count - the total number of labels in the dataset
     * @return a map of pairs of Shading and size_t indicating the likelihood
     * of each Shading feature appearing for this group with the same label
     */
    std::map<Shading, std::vector<std::vector<float>>>
    CalculateFeatureLikelihoods(
        const std::vector<Image>& class_group, size_t label_count) const;

    /**
     * Creates an empty map to hold the likelihoods of each Shading type in all
     * pixel positions for the image size this model is built for.
     * @param row_count - a size_t indicating the number of rows of pixels in 
     *                    images the model built for
     * @param column_count - a size_t indicating the number of columns of pixels 
     *                       in images the model built for
     * @return an map of Shading and 2D-vector pairs filled with zeros
     */
    static std::map<Shading, std::vector<std::vector<float>>>
    InitializeEmptyFeatureMap(size_t row_count, size_t column_count);

    /**
     * Finds the total count of each type of shading in all images in the given
     * vector of Images at the specified row and column in the image.
     * @param group - a vector of Images to find shading counts for
     * @param row - a size_t indicating the Image row index to analyze
     * @param column - a size_t indicating the Image column index to analyze
     * @return a map of pairs of Shading and size_t indicating the count of each
     * type of shading for all images in the given group of images
     */
    static std::map<Shading, size_t> CountPixelShadings(
        const std::vector<Image>& group, size_t row, size_t column);

    /**
     * Splits the Dataset object into chunks by labels and by index and creates 
     * a thread to test and classify each chunk of Images. 
     * @param dataset - a Dataset object containing Images & their actual labels
     * @param label_indices - a map of char to size_t indicating the index of 
     *                        each class label in the confusion matrix generated
     * @return a vector of thread and confusion matrix pairs
     */
    std::vector<std::pair<std::thread,
        std::future<std::vector<std::vector<size_t>>>>>
    CreateTestThreads(const Dataset& dataset,
                      const std::map<char, size_t>& label_indices) const;

    /**
     * This function is run by a single thread to classify a group of Images
     * when testing the model with multi-threading.
     * @param thread_result - a future object containing a 2D-vector that will 
     *                        be extracted after the thread is finished running
     * @param image_group - a vector of Images to Classify and validate
     * @param label_indices - a map of char to size_t indicating the index of 
     *                        each class label in the confusion matrix generated
     * @param thread_index - a size_t index of the thread running this method
     */
    void TestImageGroup(
        std::promise<std::vector<std::vector<size_t>>> thread_result,
        const std::vector<Image>& image_group,
        const std::map<char, size_t>& label_indices, size_t thread_index) const;
    
    /**
     * Collects the matrix result of each thread and creates a confusion matrix.
     * @param thread_group - a vector of thread and confusion matrix pairs
     * @param label_indices - a map of char to size_t indicating the index of 
     *                        each class label in the confusion matrix generated
     * @return a 2D-vector representing the confusion matrix for the model test
     */
    std::vector<std::vector<size_t>> JoinTestThreads(
        std::vector<std::pair<std::thread,
            std::future<std::vector<std::vector<size_t>>>>>& thread_group,
        const std::map<char, size_t>& label_indices) const;
};

} // namespace naivebayes

#endif  // NAIVE_BAYES_MODEL_H
