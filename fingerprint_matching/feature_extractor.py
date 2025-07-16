import cv2
import fingerprint_feature_extractor

def extract_and_print_features(image_path, spurious_thresh=10, invert=False, show=False, save=False):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img,
        spuriousMinutiaeThresh=spurious_thresh,
        invertImage=invert,
        showResult=show,
        saveResult=save
    )

    for feature in FeaturesTerminations:
        print(f"Termination - X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")

    for feature in FeaturesBifurcations:
        print(f"Bifurcation - X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")

    return FeaturesTerminations, FeaturesBifurcations
