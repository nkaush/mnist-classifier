#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"

#include <core/model.h>

namespace naivebayes {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 */
class NaiveBayesApp : public ci::app::App {
 public:
  NaiveBayesApp();

  void draw() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;
  void keyDown(ci::app::KeyEvent event) override;

 private:
  Sketchpad sketchpad_;
  char current_prediction_;
  
  Model model_;

  static constexpr double kWindowSize = 700;
  static constexpr double kMargin = 100;
  static constexpr size_t kImageDimension = 28;
  
  static constexpr char kDefaultImageLabel = '\0';
  
  static constexpr uint8_t kBackgroundRedIntensity = 255;
  static constexpr uint8_t kBackgroundGreenIntensity = 246;
  static constexpr uint8_t kBackgroundBlueIntensity = 148;

  static const char* kInstructionsColor;
  static const char* kPredictionColor;
  
  static const std::string kModelFilePath;
  static const std::string kUsageInstructions;
  
  static const std::string kPredictionIndicator;
};

}  // namespace visualizer

}  // namespace naivebayes
