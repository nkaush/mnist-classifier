#include <visualizer/naive_bayes_app.h>
#include <iostream>

namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin, 1) {
  std::ifstream model_file("/Users/neilkaushikkar/Cinder/my-projects/naive-bayes-nkaush/data/my-model.json");

  if (model_file.is_open()) {
    model_file >> model_;  // Deserialize the model and load it in the stack
  }
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::string(1, current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
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
      // ask your classifier to classify the image that's currently drawn on the
      // sketchpad and update current_prediction_
      current_prediction_ = model_.Classify(Image(sketchpad_.GetPixels(), '\0'));
      std::cout << current_prediction_ << std::endl;
      break;

    case ci::app::KeyEvent::KEY_BACKSPACE:
      sketchpad_.Clear();
      break;
  }
}

}  // namespace visualizer

}  // namespace naivebayes
