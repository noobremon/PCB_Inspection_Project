import cv2
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import pickle
from datetime import datetime

print("Script started")

class PCBInspectionSystem:
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_features='sqrt',
            min_samples_leaf=1,
            random_state=42
        )
        self.feature_extractor = PCBFeatureExtractor()
        self.is_trained = False
        self.model_path = "models/pcb_classifier.pkl"
        self.stats_path = "models/training_stats.json"
        
    def extract_features_from_directory(self, directory_path, label):
        """Extract features from all images in a directory"""
        features = []
        labels = []
        
        if not os.path.exists(directory_path):
            print(f"Warning: Directory {directory_path} does not exist")
            return features, labels
            
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"Processing {len(image_files)} images from {directory_path}")
        
        for filename in image_files:
            image_path = os.path.join(directory_path, filename)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {filename}")
                    continue
                    
                feature_vector = self.feature_extractor.extract_features(image)
                if feature_vector is not None and len(feature_vector) > 0:
                    features.append(feature_vector)
                    labels.append(label)
                    print(f"Processed: {filename}")
                else:
                    print(f"Warning: Could not extract features from {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
        return features, labels
    
    def train_model(self, good_pcb_dir="dataset/good", defective_pcb_dir="dataset/defective"):
        """Train the classification model"""
        print("Starting model training...")
        
        # Extract features from good PCBs
        good_features, good_labels = self.extract_features_from_directory(good_pcb_dir, 0)  # 0 for good
        
        # Extract features from defective PCBs
        defective_features, defective_labels = self.extract_features_from_directory(defective_pcb_dir, 1)  # 1 for defective
        
        if len(good_features) == 0 and len(defective_features) == 0:
            print("Error: No training data found. Please add images to the dataset directories.")
            return False
        
        if len(good_features) == 0:
            print("Error: No good PCB images found. Please add images to the 'good' directory.")
            return False
            
        if len(defective_features) == 0:
            print("Error: No defective PCB images found. Please add images to the 'defective' directory.")
            return False
        
        # Combine all features and labels
        all_features = good_features + defective_features
        all_labels = good_labels + defective_labels
        
        print(f"Total training samples: {len(all_features)} (Good: {len(good_features)}, Defective: {len(defective_features)})")
        
        # Check if we have enough samples for both classes
        if len(good_features) < 2 or len(defective_features) < 2:
            print("Warning: Very few samples for training. Consider adding more images.")
            if len(all_features) < 4:
                print("Error: Not enough samples for proper train/test split. Need at least 4 samples total.")
                return False
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Use stratified split to ensure both classes are represented in train and test sets
        try:
            if len(np.unique(y)) == 2 and len(all_features) >= 4:
                # Use stratified split when we have both classes and enough samples
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(sss.split(X, y))
                x_train, x_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            else:
                # Fallback to regular split
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error in data splitting: {str(e)}")
            # Use all data for training if splitting fails
            x_train, y_train = X, y
            x_test, y_test = X, y
        
        try:
            # Train the classifier
            self.classifier.fit(x_train, y_train)
            
            # Validate the model
            y_pred = self.classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print("Model trained successfully!")
            print(f"Validation Accuracy: {accuracy:.2f}")
            
            # Generate classification report only if we have both classes in test set
            unique_test_labels = np.unique(y_test)
            unique_pred_labels = np.unique(y_pred)
            
            if len(unique_test_labels) > 1:
                print("\nClassification Report:")
                target_names = ['Good', 'Defective']
                print(classification_report(y_test, y_pred, target_names=target_names))
            else:
                print(f"\nNote: Test set only contains class: {'Good' if unique_test_labels[0] == 0 else 'Defective'}")
                print("Classification report skipped due to single class in test set.")
            
            # Save the model
            self.save_model()
            
            # Save training statistics
            stats = {
                'training_date': datetime.now().isoformat(),
                'total_samples': len(all_features),
                'good_samples': len(good_features),
                'defective_samples': len(defective_features),
                'validation_accuracy': float(accuracy),
                'feature_count': len(all_features[0]) if all_features else 0,
                'classes_in_training': len(np.unique(y)),
                'classes_in_test': len(unique_test_labels)
            }
            
            os.makedirs("models", exist_ok=True)
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return False
    
    def save_model(self):
        """Save the trained model"""
        try:
            os.makedirs("models", exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a pre-trained model"""
        if not os.path.exists(self.model_path):
            print("No trained model found. Please train the model first.")
            return False
            
        try:
            # Check if file is not empty
            if os.path.getsize(self.model_path) == 0:
                print("Model file is empty. Please retrain the model.")
                return False
                
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.is_trained = True
            print("Model loaded successfully")
            return True
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading model: {str(e)}")
            print("The model file appears to be corrupted. Please retrain the model.")
            return False
        except Exception as e:
            print(f"Unexpected error loading model: {str(e)}")
            return False
    
    def inspect_pcb(self, image_path):
        """Inspect a single PCB image"""
        if not self.is_trained and not self.load_model():
            print("Error: No trained model available. Please train the model first.")
            return None
        
        try:
            # Load and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return None
            
            # Extract features
            features = self.feature_extractor.extract_features(image)
            if features is None or len(features) == 0:
                print("Error: Could not extract features from the image")
                return None
            
            # Make prediction
            prediction = self.classifier.predict([features])[0]
            confidence = max(self.classifier.predict_proba([features])[0])
            
            result = {
                'image_path': image_path,
                'prediction': 'Good' if prediction == 0 else 'Defective',
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error inspecting PCB: {str(e)}")
            return None
    
    def batch_inspect(self, test_directory="test_images"):
        """Inspect all images in a directory"""
        if not os.path.exists(test_directory):
            print(f"Test directory {test_directory} does not exist")
            return []
        
        results = []
        image_files = [f for f in os.listdir(test_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {test_directory}")
            return []
        
        print(f"Inspecting {len(image_files)} images...")
        
        for filename in image_files:
            image_path = os.path.join(test_directory, filename)
            result = self.inspect_pcb(image_path)
            if result:
                results.append(result)
                print(f"{filename}: {result['prediction']} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"Failed to process {filename}")
        
        return results


class PCBFeatureExtractor:
    def __init__(self):
        self.target_size = (256, 256)
    
    def extract_features(self, image):
        """Extract comprehensive features from PCB image"""
        try:
            # Resize image to standard size
            resized = cv2.resize(image, self.target_size)
            
            # Convert to different color spaces
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            features = []
            
            # 1. Statistical features
            features.extend(self._statistical_features(gray))
            
            # 2. Texture features using Local Binary Pattern
            features.extend(self._texture_features(gray))
            
            # 3. Edge features
            features.extend(self._edge_features(gray))
            
            # 4. Color features
            features.extend(self._color_features(resized, hsv))
            
            # 5. Contour features
            features.extend(self._contour_features(gray))
            
            # 6. Frequency domain features
            features.extend(self._frequency_features(gray))
            
            # Ensure all features are finite numbers
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def _statistical_features(self, gray):
        """Extract statistical features"""
        features = []
        try:
            features.append(np.mean(gray))
            features.append(np.std(gray))
            features.append(np.var(gray))
            features.append(np.min(gray))
            features.append(np.max(gray))
            features.append(np.median(gray))
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            if hist.sum() > 0:
                hist = hist / hist.sum()  # Normalize
            features.extend(hist[::16])  # Sample every 16th bin to reduce dimensionality
            
        except Exception as e:
            print(f"Error in statistical features: {str(e)}")
            features = [0.0] * 22  # Return zeros if calculation fails
            
        return features
    
    def _texture_features(self, gray):
        """Extract texture features using Local Binary Pattern"""
        features = []
        try:
            # Simple LBP implementation
            lbp = self._local_binary_pattern(gray, 8, 1)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = lbp_hist.flatten()
            if lbp_hist.sum() > 0:
                lbp_hist = lbp_hist / lbp_hist.sum()
            features.extend(lbp_hist[::16])  # Sample every 16th bin
            
        except Exception as e:
            print(f"Error in texture features: {str(e)}")
            features = [0.0] * 16  # Return zeros if calculation fails
            
        return features
    
    def _local_binary_pattern(self, image, num_points, radius):
        """Simple Local Binary Pattern implementation"""
        try:
            height, width = image.shape
            lbp = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    lbp[i, j] = self._calculate_lbp_pixel(image, i, j, num_points, radius, height, width)
            
            return lbp
        except Exception:
            return np.zeros_like(image)

    def _calculate_lbp_pixel(self, image, i, j, num_points, radius, height, width):
        """Helper to calculate LBP value for a single pixel"""
        try:
            center = image[i, j]
            binary = []
            for k in range(num_points):
                angle = 2 * np.pi * k / num_points
                x = int(j + radius * np.cos(angle))
                y = int(i - radius * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    binary.append(1 if image[y, x] >= center else 0)
                else:
                    binary.append(0)
            return sum([binary[k] * (2 ** k) for k in range(num_points)])
        except Exception:
            return 0
    
    def _edge_features(self, gray):
        """Extract edge-based features"""
        features = []
        try:
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            features.append(np.sum(edges) / (edges.shape[0] * edges.shape[1]))  # Edge density
            
            # Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            features.append(np.mean(np.abs(grad_x)))
            features.append(np.mean(np.abs(grad_y)))
            features.append(np.std(grad_x))
            features.append(np.std(grad_y))
            
        except Exception as e:
            print(f"Error in edge features: {str(e)}")
            features = [0.0] * 5  # Return zeros if calculation fails
            
        return features
    
    def _color_features(self, bgr_image, hsv_image):
        """Extract color-based features"""
        features = []
        try:
            # BGR channel statistics
            for channel in range(3):
                features.append(np.mean(bgr_image[:, :, channel]))
                features.append(np.std(bgr_image[:, :, channel]))
            
            # HSV channel statistics
            for channel in range(3):
                features.append(np.mean(hsv_image[:, :, channel]))
                features.append(np.std(hsv_image[:, :, channel]))
                
        except Exception as e:
            print(f"Error in color features: {str(e)}")
            features = [0.0] * 12  # Return zeros if calculation fails
            
        return features
    
    def _contour_features(self, gray):
        """Extract contour-based features"""
        features = []
        try:
            # Apply threshold to get binary image for better contour detection
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Number of contours
                features.append(len(contours))
                
                # Area statistics
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                features.append(np.mean(areas) if areas else 0)
                features.append(np.std(areas) if areas else 0)
                features.append(max(areas) if areas else 0)
                
                # Perimeter statistics
                perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10]
                features.append(np.mean(perimeters) if perimeters else 0)
                features.append(np.std(perimeters) if perimeters else 0)
            else:
                features.extend([0] * 6)
                
        except Exception as e:
            print(f"Error in contour features: {str(e)}")
            features = [0.0] * 6  # Return zeros if calculation fails
            
        return features
    
    def _frequency_features(self, gray):
        """Extract frequency domain features using FFT"""
        features = []
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Statistical features of magnitude spectrum
            features.append(np.mean(magnitude_spectrum))
            features.append(np.std(magnitude_spectrum))
            features.append(np.max(magnitude_spectrum))
            
        except Exception as e:
            print(f"Error in frequency features: {str(e)}")
            features = [0.0] * 3  # Return zeros if calculation fails
            
        return features


def main():
    """Main function to demonstrate the PCB inspection system"""
    inspector = PCBInspectionSystem()
    print("PCB Automated Optical Inspection System")
    print("=" * 40)
    
    menu_options = {
        '1': lambda: train_model_option(inspector),
        '2': lambda: inspect_single_image_option(inspector),
        '3': lambda: batch_inspect_option(inspector),
        '4': lambda: view_training_stats_option(inspector),
        '5': exit_option
    }
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        action = menu_options.get(choice, invalid_choice_option)
        if action() == "exit":
            break

def print_menu():
    print("\nOptions:")
    print("1. Train the model")
    print("2. Inspect a single PCB image")
    print("3. Batch inspect images")
    print("4. View training statistics")
    print("5. Exit")

def train_model_option(inspector):
    print("\nTraining the model...")
    
    # Check if dataset directories exist
    good_dir = "dataset/good"
    defective_dir = "dataset/defective"
    
    if not os.path.exists(good_dir):
        print(f"Creating directory: {good_dir}")
        os.makedirs(good_dir, exist_ok=True)
        
    if not os.path.exists(defective_dir):
        print(f"Creating directory: {defective_dir}")
        os.makedirs(defective_dir, exist_ok=True)
    
    # Check if directories have images
    good_images = len([f for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    defective_images = len([f for f in os.listdir(defective_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    if good_images == 0 or defective_images == 0:
        print("\nDataset Setup Required:")
        print(f"Good PCB images: {good_images} (in {good_dir})")
        print(f"Defective PCB images: {defective_images} (in {defective_dir})")
        print("\nPlease add images to both directories before training.")
        return
    
    success = inspector.train_model()
    if success:
        print("Model training completed successfully!")
    else:
        print("Model training failed. Please check your dataset and try again.")

def inspect_single_image_option(inspector):
    image_path = input("Enter the path to the PCB image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return
        
    result = inspector.inspect_pcb(image_path)
    if result:
        print("\nInspection Result:")
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print("Failed to inspect the image.")

def batch_inspect_option(inspector):
    test_dir = input("Enter test directory path (or press Enter for 'test_images'): ").strip()
    if not test_dir:
        test_dir = "test_images"
        
    if not os.path.exists(test_dir):
        print(f"Creating test directory: {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
        print(f"Please add images to {test_dir} and try again.")
        return
        
    results = inspector.batch_inspect(test_dir)
    print(f"\nBatch inspection completed. Processed {len(results)} images.")
    
    if results:
        # Save results
        results_file = f"results/inspection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Show summary
        good_count = sum(1 for r in results if r['prediction'] == 'Good')
        defective_count = len(results) - good_count
        print(f"Summary: {good_count} Good, {defective_count} Defective")

def view_training_stats_option(inspector):
    if os.path.exists(inspector.stats_path):
        with open(inspector.stats_path, 'r') as f:
            stats = json.load(f)
        print("\nTraining Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("No training statistics available. Train the model first.")

def exit_option():
    print("Exiting...")
    return "exit"

def invalid_choice_option():
    print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()