import cv2
import fingerprint_feature_extractor


if __name__ == '__main__':
    img = cv2.imread('enhanced/3.jpg', 0)	# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage = False, showResult=True, saveResult = True)
    
    
    for feature in FeaturesTerminations:  # or FeaturesTerminations
        print(f"X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")

    for feature in FeaturesBifurcations:  # or FeaturesBifurcations
        print(f"X: {feature.locX}, Y: {feature.locY}, Angle: {feature.Orientation}, Type: {feature.Type}")


# from test import extract_and_print_features

# test=extract_and_print_features()


