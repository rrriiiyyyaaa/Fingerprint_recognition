import cv2
import fingerprint_feature_extractor  # Make sure this module is available
from  feature_extractor import *

def extract_and_print_features(image_path: str):

    img = cv2.imread(image_path, 0)  # Read in grayscale
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img,
        spuriousMinutiaeThresh=10,
        invertImage=False,
        showResult=True,
        saveResult=True
    )

    
