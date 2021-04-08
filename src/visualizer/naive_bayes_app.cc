#include <visualizer/naive_bayes_app.h>
#include <iostream>

namespace naivebayes {

namespace visualizer {

const std::string NaiveBayesApp::kModelFilePath = 
    "/Users/neilkaushikkar/Cinder/my-projects/"
    "naive-bayes-nkaush/data/cinder-app-model.json";
const std::string NaiveBayesApp::kUsageInstructions = 
    "Press Delete to clear the sketchpad. Press Enter to make a prediction.";
const std::string NaiveBayesApp::kPredictionIndicator = "Prediction: ";

const char* NaiveBayesApp::kInstructionsColor = "black";
const char* NaiveBayesApp::kPredictionColor = "blue";

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin, 1) {
  std::ifstream model_file(kModelFilePath);

  if (model_file.is_open()) {
    model_file >> model_;  // Deserialize the model and load it in the stack
  }
  
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(kBackgroundRedIntensity, 
                               kBackgroundGreenIntensity, 
                               kBackgroundBlueIntensity);
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      kUsageInstructions, glm::vec2(kWindowSize / 2, kMargin / 2), 
      ci::Color(kInstructionsColor)); // to center, find the middle

  ci::gl::drawStringCentered(
      kPredictionIndicator + std::string(1, current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), 
      ci::Color(kPredictionColor)); // to center, find the middle
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN:
      // Make a new Image with the pixels without specifying what label it is
      current_prediction_ = model_.Classify(Image(sketchpad_.GetPixels(), 
                                                  kDefaultImageLabel));
      break;

    case ci::app::KeyEvent::KEY_BACKSPACE:
      sketchpad_.Clear();
      break;
  }
}

}  // namespace visualizer

}  // namespace naivebayes
