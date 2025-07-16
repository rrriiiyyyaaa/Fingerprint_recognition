import cv2
import fingerprint_feature_extractor  # Make sure this module is available

def extract_and_print_features():

    img = cv2.imread('enhanced/3.jpg', 0)  # Read in grayscale
    if img is None:
        print(f"Failed to load image: {'enhanced/3.jpg'}")
        return

    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img,
        spuriousMinutiaeThresh=10,
        invertImage=False,
        showResult=True,
        saveResult=True
    )

    print("\n--- Termination Points ---")
    for feature in FeaturesTerminations:
        print(f"X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")

    print("\n--- Bifurcation Points ---")
    for feature in FeaturesBifurcations:
        print(f"X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")