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
        self.defect_detector = PCBDefectDetector()
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
            
            # If defective, show visual analysis
            if prediction == 1:  # Defective
                self.show_defect_analysis(image, image_path, result)
            
            return result
            
        except Exception as e:
            print(f"Error inspecting PCB: {str(e)}")
            return None
    
    def show_defect_analysis(self, image, image_path, result):
        """Show defect analysis in a popup window"""
        try:
            print("Displaying defect analysis...")
            defect_image = self.defect_detector.detect_and_highlight_defects(image)
            
            # Create a display window
            window_name = f"PCB Defect Analysis - {os.path.basename(image_path)}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Add text overlay with prediction info
            display_image = defect_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            text1 = f"Prediction: {result['prediction']}"
            text2 = f"Confidence: {result['confidence']:.2f}"
            text3 = "Press any key to close"
            
            cv2.putText(display_image, text1, (10, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(display_image, text2, (10, 70), font, 1, (0, 0, 255), 2)
            cv2.putText(display_image, text3, (10, display_image.shape[0] - 20), font, 0.7, (255, 255, 255), 2)
            
            # Resize image for display if it's too large
            height, width = display_image.shape[:2]
            if height > 800 or width > 1200:
                scale = min(800/height, 1200/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            cv2.imshow(window_name, display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error displaying defect analysis: {str(e)}")
    
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
    
    def debug_defect_detection(self, image_path):
        """Debug method to understand why classification and detection don't match"""
        if not self.is_trained and not self.load_model():
            print("Error: No trained model available.")
            return

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return

            print(f"=== DEBUG ANALYSIS FOR: {image_path} ===")

            # 1. Get classification result
            features = self.feature_extractor.extract_features(image)
            prediction = self.classifier.predict([features])[0]
            probabilities = self.classifier.predict_proba([features])[0]

            print(f"Classification: {'Defective' if prediction == 1 else 'Good'}")
            print(f"Probabilities: Good={probabilities[0]:.3f}, Defective={probabilities[1]:.3f}")

            # 2. Analyze the extracted features
            print(f"\nFeature Analysis:")
            print(f"Total features extracted: {len(features)}")

            # Get feature importance from Random Forest
            feature_importance = self.classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]  # Top 10 most important features

            print(f"Top 10 most important features:")
            for i, idx in enumerate(reversed(top_features)):
                print(f"{i+1}. Feature {idx}: {features[idx]:.3f} (importance: {feature_importance[idx]:.3f})")

            # 3. Manual defect detection with debug info
            print(f"\nDefect Detection Analysis:")

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 15, 35, 35)
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

            # Test different threshold values
            mean_intensity = np.mean(blurred)
            std_intensity = np.std(blurred)

            print(f"Image statistics:")
            print(f"  Mean intensity: {mean_intensity:.2f}")
            print(f"  Std intensity: {std_intensity:.2f}")
            print(f"  Min intensity: {np.min(blurred):.2f}")
            print(f"  Max intensity: {np.max(blurred):.2f}")

            # Check for potential defects with lower thresholds
            print(f"\nTesting with relaxed thresholds:")

            # Test bright defects with lower threshold
            bright_threshold_relaxed = mean_intensity + 2 * std_intensity
            _, bright_mask = cv2.threshold(blurred, bright_threshold_relaxed, 255, cv2.THRESH_BINARY)
            bright_pixels = np.sum(bright_mask > 0)
            print(f"  Bright anomalies (threshold {bright_threshold_relaxed:.0f}): {bright_pixels} pixels")

            # Test dark defects with higher threshold
            dark_threshold_relaxed = mean_intensity - 2 * std_intensity
            _, dark_mask = cv2.threshold(blurred, dark_threshold_relaxed, 255, cv2.THRESH_BINARY_INV)
            dark_pixels = np.sum(dark_mask > 0)
            print(f"  Dark anomalies (threshold {dark_threshold_relaxed:.0f}): {dark_pixels} pixels")

            # Check edge density
            edges = cv2.Canny(blurred, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            print(f"  Edge density: {edge_density:.4f}")

            # 4. Show debug images
            self.show_debug_images(image, gray, bright_mask, dark_mask, edges, image_path)

            # 5. Suggestions
            print(f"\nSuggestions:")
            if prediction == 1 and bright_pixels == 0 and dark_pixels == 0:
                print("  - The defect might be very subtle or microscopic")
                print("  - Consider lowering min_defect_area threshold")
                print("  - The defect might be in texture/color rather than brightness")
                print("  - Check for component misalignment or surface finish issues")

        except Exception as e:
            print(f"Error in debug analysis: {str(e)}")

def show_debug_images(self, original, gray, bright_mask, dark_mask, edges, image_path):
    """Show debug images side by side"""
    try:
        # Create a combined debug image
        h, w = gray.shape
        debug_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Original
        debug_image[0:h, 0:w] = cv2.resize(original, (w, h))
        
        # Top-right: Grayscale
        debug_image[0:h, w:2*w] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Bottom-left: Bright mask
        debug_image[h:2*h, 0:w] = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
        
        # Bottom-right: Dark mask
        debug_image[h:2*h, w:2*w] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_image, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "Grayscale", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "Bright Mask", (10, h + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, "Dark Mask", (w + 10, h + 30), font, 0.7, (255, 255, 255), 2)
        
        # Show the debug image
        window_name = f"Debug Analysis - {os.path.basename(image_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error showing debug images: {str(e)}")

# Modified defect detection with relaxed mode
def detect_with_relaxed_thresholds(self, image):
    """Try defect detection with more relaxed thresholds"""
    try:
        # Temporarily reduce thresholds
        original_min_area = self.defect_detector.min_defect_area
        original_bright_threshold = self.defect_detector.bright_threshold
        original_dark_threshold = self.defect_detector.dark_threshold
        
        # Use more relaxed thresholds
        self.defect_detector.min_defect_area = 100  # Much smaller
        
        # Convert to grayscale and get adaptive thresholds
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        self.defect_detector.bright_threshold = mean_intensity + 1.5 * std_intensity
        self.defect_detector.dark_threshold = mean_intensity - 1.5 * std_intensity
        
        print(f"Using relaxed thresholds:")
        print(f"  Min area: {self.defect_detector.min_defect_area}")
        print(f"  Bright threshold: {self.defect_detector.bright_threshold:.0f}")
        print(f"  Dark threshold: {self.defect_detector.dark_threshold:.0f}")
        
        # Detect defects with relaxed thresholds
        result_image = self.defect_detector.detect_and_highlight_defects(image)
        
        # Restore original thresholds
        self.defect_detector.min_defect_area = original_min_area
        self.defect_detector.bright_threshold = original_bright_threshold
        self.defect_detector.dark_threshold = original_dark_threshold
        
        return result_image
        
    except Exception as e:
        print(f"Error in relaxed detection: {str(e)}")
        return image


class PCBDefectDetector:
    def __init__(self):
        # Adjusted parameters for better detection of subtle defects
        self.gaussian_kernel_size = 3
        self.min_defect_area = 15    # Reduced to detect smaller defects
        self.max_defect_area = 5000  # Maximum area to avoid large false positives
        self.bright_threshold = 200  # Lowered to detect more subtle bright defects
        self.dark_threshold = 60     # Increased to detect more subtle dark defects
        self.edge_threshold_low = 80  # Adjusted for better edge detection
        self.edge_threshold_high = 180
        self.debug_mode = True  # Enable debug mode for more detailed logging
        
    def detect_and_highlight_defects(self, image):
        """Detect and highlight only actual defects with improved accuracy"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
            
            # Create a copy of the original image for highlighting
            result_image = image.copy()
            
            # Detect defects using multiple methods with stricter criteria
            defect_regions = []
            
            # Method 1: Detect significant bright anomalies (soldering issues)
            bright_defects = self._detect_bright_defects(blurred)
            defect_regions.extend(bright_defects)
            
            # Method 2: Detect significant dark anomalies (missing components)
            dark_defects = self._detect_dark_defects(blurred)
            defect_regions.extend(dark_defects)
            
            # Method 3: Detect structural anomalies using edge analysis
            structural_defects = self._detect_structural_defects(blurred)
            defect_regions.extend(structural_defects)
            
            # Method 4: Detect surface defects (scratches, burns)
            surface_defects = self._detect_surface_defects(blurred)
            defect_regions.extend(surface_defects)
            
            # Remove overlapping defects and filter by size
            filtered_defects = self._filter_and_merge_defects(defect_regions)
            
            # Highlight only the filtered defects
            defect_count = 0
            for defect in filtered_defects:
                x, y, w, h, defect_type = defect
                
                # Different colors for different defect types
                color_map = {
                    'bright': (0, 255, 255),    # Yellow for bright defects
                    'dark': (255, 0, 0),        # Blue for dark defects
                    'structural': (0, 0, 255),  # Red for structural defects
                    'surface': (255, 0, 255)    # Magenta for surface defects
                }
                
                color = color_map.get(defect_type, (0, 0, 255))
                
                # Draw rectangle around defect
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw circle at center
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(result_image, (center_x, center_y), 10, color, 2)
                
                # Add defect label
                label = f"{defect_type.upper()}"
                cv2.putText(result_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                defect_count += 1
            
            # Add defect count to image
            cv2.putText(result_image, f"Defects Found: {defect_count}", 
                       (10, result_image.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return result_image
            
        except Exception as e:
            print(f"Error in defect detection: {str(e)}")
            return image
    
    def _detect_bright_defects(self, gray):
        """Detect bright defects like excess solder or burn marks with improved sensitivity"""
        defects = []
        try:
            if self.debug_mode:
                print("Detecting bright defects...")
            
            # Use adaptive thresholding to handle varying lighting conditions
            bright_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply additional global threshold to ensure we get the brightest regions
            _, global_mask = cv2.threshold(gray, self.bright_threshold, 255, cv2.THRESH_BINARY)
            bright_mask = cv2.bitwise_and(bright_mask, global_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.debug_mode:
                print(f"Found {len(contours)} potential bright regions")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                    
                # Get bounding box and calculate features
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / max(h, 1)  # Avoid division by zero
                
                # Extract region of interest
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                    
                # Calculate features for filtering
                mean_intensity = np.mean(roi)
                intensity_std = np.std(roi)
                
                # More sophisticated filtering
                if (0.2 <= aspect_ratio <= 5.0 and  # Allow slightly more aspect ratio variation
                    intensity_std < 40 and           # Slightly higher std threshold
                    mean_intensity > self.bright_threshold):  # Ensure it's actually bright
                    
                    # Check contrast with surrounding area
                    roi_larger = gray[max(0,y-5):min(gray.shape[0],y+h+5), 
                                    max(0,x-5):min(gray.shape[1],x+w+5)]
                    if roi_larger.size > 0:
                        contrast = mean_intensity - np.mean(roi_larger)
                        if contrast > 15:  # Minimum contrast threshold
                            defects.append((x, y, w, h, 'bright'))
                            
                            if self.debug_mode:
                                print(f"  - Bright defect {i+1}: area={area}, mean={mean_intensity:.1f}, "
                                      f"std={intensity_std:.1f}, aspect={aspect_ratio:.1f}, contrast={contrast:.1f}")
            
            if self.debug_mode:
                print(f"Total bright defects found: {len(defects)}")
                
        except Exception as e:
            print(f"Error detecting bright defects: {str(e)}")
            
        return defects
    
    def _detect_dark_defects(self, gray):
        """Detect dark defects like missing components or holes with improved sensitivity"""
        defects = []
        try:
            if self.debug_mode:
                print("Detecting dark defects...")
            
            # Use adaptive thresholding to handle varying lighting
            dark_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply additional global threshold to ensure we get the darkest regions
            _, global_mask = cv2.threshold(gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, global_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.debug_mode:
                print(f"Found {len(contours)} potential dark regions")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                    
                # Get bounding box and calculate features
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / max(h, 1)  # Avoid division by zero
                
                # Extract region of interest
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                    
                # Calculate features for filtering
                mean_intensity = np.mean(roi)
                intensity_std = np.std(roi)
                
                # Get surrounding area for contrast calculation
                margin = min(10, min(w, h))  # Dynamic margin based on defect size
                roi_larger = gray[max(0,y-margin):min(gray.shape[0],y+h+margin), 
                                max(0,x-margin):min(gray.shape[1],x+w+margin)]
                
                if roi_larger.size > 0:
                    contrast = np.mean(roi_larger) - mean_intensity
                    
                    # More sophisticated filtering
                    if (0.2 <= aspect_ratio <= 5.0 and  # Allow more aspect ratio variation
                        contrast > 20 and               # Minimum contrast threshold (reduced from 50)
                        mean_intensity < self.dark_threshold + 30):  # Not too bright
                        
                        # Additional check for consistent darkness
                        if intensity_std < 40:  # Consistent darkness
                            defects.append((x, y, w, h, 'dark'))
                            
                            if self.debug_mode:
                                print(f"  - Dark defect {i+1}: area={area}, mean={mean_intensity:.1f}, "
                                      f"std={intensity_std:.1f}, aspect={aspect_ratio:.1f}, contrast={contrast:.1f}")
            
            if self.debug_mode:
                print(f"Total dark defects found: {len(defects)}")
                
        except Exception as e:
            print(f"Error detecting dark defects: {str(e)}")
            
        return defects
    
    def _filter_duplicate_defects(self, defects, overlap_threshold=0.5):
        """Filter out duplicate or highly overlapping defects"""
        if not defects:
            return []
            
        # Convert to numpy array for easier manipulation
        defects_array = np.array([(x, y, x+w, y+h, t) for x, y, w, h, t in defects])
        
        # Calculate areas
        areas = (defects_array[:, 2] - defects_array[:, 0]) * (defects_array[:, 3] - defects_array[:, 1])
        
        # Sort by area (descending)
        sorted_indices = np.argsort(-areas)
        keep = [True] * len(defects)
        
        for i in range(len(sorted_indices)):
            if not keep[sorted_indices[i]]:
                continue
                
            for j in range(i + 1, len(sorted_indices)):
                if not keep[sorted_indices[j]]:
                    continue
                    
                # Calculate intersection over union
                x1 = max(defects_array[sorted_indices[i], 0], defects_array[sorted_indices[j], 0])
                y1 = max(defects_array[sorted_indices[i], 1], defects_array[sorted_indices[j], 1])
                x2 = min(defects_array[sorted_indices[i], 2], defects_array[sorted_indices[j], 2])
                y2 = min(defects_array[sorted_indices[i], 3], defects_array[sorted_indices[j], 3])
                
                if x2 <= x1 or y2 <= y1:
                    continue  # No overlap
                    
                intersection = (x2 - x1) * (y2 - y1)
                area_i = areas[sorted_indices[i]]
                area_j = areas[sorted_indices[j]]
                union = area_i + area_j - intersection
                
                if intersection / min(area_i, area_j) > overlap_threshold:
                    # Keep the larger defect
                    if area_i > area_j:
                        keep[sorted_indices[j]] = False
                    else:
                        keep[sorted_indices[i]] = False
                        break
                        
        return [defects[i] for i in range(len(defects)) if keep[i]]
    
    def _detect_structural_defects(self, gray):
        """Detect structural defects using edge analysis with improved sensitivity"""
        defects = []
        try:
            if self.debug_mode:
                print("Detecting structural defects...")
                
            # Apply adaptive thresholding to handle varying lighting
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply Canny edge detection with adjusted thresholds
            edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
            
            # Combine with thresholded image to enhance edge detection
            edges = cv2.bitwise_or(edges, thresh)
            
            # Apply morphological operations to connect broken edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.debug_mode:
                print(f"Found {len(contours)} potential structural defects")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                    
                # Calculate convex hull to check for irregular shapes
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Calculate circularity
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                    
                    # More sophisticated filtering
                    if (solidity < 0.7 or  # Irregular shape
                        (circularity < 0.3 and area > 100) or  # Very non-circular and reasonably large
                        (circularity > 0.9 and area > 200)):   # Very circular but large (potential hole)
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        defect_type = 'structural'
                        
                        # Classify based on shape
                        if circularity > 0.7:
                            defect_type = 'hole' if area > 100 else 'spot'
                        elif w > 2 * h or h > 2 * w:
                            defect_type = 'crack' if area > 50 else 'scratch'
                            
                        defects.append((x, y, w, h, defect_type))
                        
                        if self.debug_mode:
                            print(f"  - {defect_type.capitalize()} defect {i+1}: area={area}, "
                                  f"solidity={solidity:.2f}, circularity={circularity:.2f}")
            
            if self.debug_mode:
                print(f"Total structural defects found: {len(defects)}")
                            
        except Exception as e:
            print(f"Error detecting structural defects: {str(e)}")
            
        return defects
    
    def _detect_surface_defects(self, gray):
        """Detect surface defects using multiple complementary methods"""
        defects = []
        try:
            if self.debug_mode:
                print("\n=== Starting surface defect detection ===")
            
            # Method 1: Enhanced texture-based detection
            if self.debug_mode:
                print("\nRunning texture-based detection...")
            texture_defects = self._detect_texture_defects(gray)
            if self.debug_mode:
                print(f"Found {len(texture_defects)} texture defects")
            defects.extend(texture_defects)
            
            # Method 2: Enhanced scratch detection using line detection
            if self.debug_mode:
                print("\nRunning scratch detection...")
            scratch_defects = self._detect_scratch_defects(gray)
            if self.debug_mode:
                print(f"Found {len(scratch_defects)} scratch defects")
            defects.extend(scratch_defects)
            
            # Method 3: Directional gradient analysis for linear defects
            if self.debug_mode:
                print("\nRunning gradient-based detection...")
            gradient_defects = self._detect_gradient_defects(gray)
            if self.debug_mode:
                print(f"Found {len(gradient_defects)} gradient-based defects")
            defects.extend(gradient_defects)
            
            # Remove duplicate detections (overlapping regions)
            if len(defects) > 1:
                defects = self._filter_duplicate_defects(defects)
            
            if self.debug_mode:
                print(f"\n=== Total surface defects found: {len(defects)} ===")
                
        except Exception as e:
            print(f"\nError in surface defect detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return defects

def _detect_texture_defects(self, gray):
    """Original texture-based defect detection"""
    defects = []
    try:
        # Apply local standard deviation filter
        kernel = np.ones((9, 9), np.float32) / 81
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        std_dev = np.sqrt(sqr_mean - mean**2)
        
        # Threshold for high texture variation
        _, texture_mask = cv2.threshold(std_dev.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_defect_area <= area <= self.max_defect_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the texture variation is consistent (not noise)
                roi = std_dev[y:y+h, x:x+w]
                if np.mean(roi) > 35:  # Consistent high variation
                    defects.append((x, y, w, h, 'surface'))
                    
    except Exception as e:
        print(f"Error detecting texture defects: {str(e)}")
        
    return defects

def _detect_scratch_defects(self, gray):
    """Enhanced scratch detection using line detection and morphological operations"""
    defects = []
    try:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Calculate gradients in both directions
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude * 255 / np.max(gradient_magnitude))
        
        # Threshold gradient magnitude to find strong edges
        _, edge_mask = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # Use morphological operations to detect linear features
        # Horizontal line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical line detection
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, vertical_kernel)
        
        # Diagonal line detection (45 degrees)
        diagonal_kernel1 = np.array([[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]], dtype=np.uint8)
        diagonal_lines1 = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, diagonal_kernel1)
        
        # Diagonal line detection (-45 degrees)
        diagonal_kernel2 = np.array([[0, 0, 0, 0, 1],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [1, 0, 0, 0, 0]], dtype=np.uint8)
        diagonal_lines2 = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, diagonal_kernel2)
        
        # Combine all line detections
        lines_combined = cv2.bitwise_or(horizontal_lines, vertical_lines)
        lines_combined = cv2.bitwise_or(lines_combined, diagonal_lines1)
        lines_combined = cv2.bitwise_or(lines_combined, diagonal_lines2)
        
        # Apply additional morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        lines_combined = cv2.morphologyEx(lines_combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(lines_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 50:  # Lower threshold for linear features
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio for linear features
                aspect_ratio = max(w, h) / min(w, h)
                
                # Accept features with high aspect ratio (linear scratches)
                if aspect_ratio >= 3.0:  # Linear feature
                    # Additional validation: check if it's a real scratch
                    roi = gray[y:y+h, x:x+w]
                    
                    # Check for intensity variation along the scratch
                    if w > h:  # Horizontal scratch
                        profile = np.mean(roi, axis=0)
                    else:  # Vertical scratch
                        profile = np.mean(roi, axis=1)
                    
                    # Check if there's a consistent intensity drop/rise
                    if np.std(profile) > 10:  # Significant variation
                        defects.append((x, y, w, h, 'scratch'))
                        
    except Exception as e:
        print(f"Error detecting scratch defects: {str(e)}")
        
    return defects

def _detect_gradient_defects(self, gray):
    """Detect defects using gradient-based analysis for subtle variations"""
    defects = []
    try:
        if self.debug_mode:
            print("Running gradient-based defect detection...")
        
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Normalize gradient magnitude to 0-255 range
        gradient_magnitude = np.uint8(gradient_magnitude * 255 / np.max(gradient_magnitude))
        
        # Use adaptive threshold for gradient magnitude
        grad_thresh = cv2.adaptiveThreshold(
            gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        grad_thresh = cv2.morphologyEx(grad_thresh, cv2.MORPH_CLOSE, kernel)
        grad_thresh = cv2.morphologyEx(grad_thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(grad_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Lower the minimum area threshold to catch smaller defects
            if area >= 25:  # Reduced from typical min_defect_area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract ROI for further analysis
                roi = gray[y:y+h, x:x+w]
                roi_grad = gradient_magnitude[y:y+h, x:x+w]
                
                if roi.size > 0 and roi_grad.size > 0:
                    # Calculate statistics
                    mean_intensity = np.mean(roi)
                    std_intensity = np.std(roi)
                    mean_gradient = np.mean(roi_grad)
                    
                    # Check for significant gradient variation (indicates defect)
                    if mean_gradient > 30 and std_intensity > 8:
                        # Additional validation: check contrast with surrounding area
                        margin = 5
                        y_start = max(0, y - margin)
                        y_end = min(gray.shape[0], y + h + margin)
                        x_start = max(0, x - margin)
                        x_end = min(gray.shape[1], x + w + margin)
                        
                        surrounding = gray[y_start:y_end, x_start:x_end]
                        if surrounding.size > roi.size:
                            # Remove the defect area from surrounding calculation
                            surrounding_mask = np.ones(surrounding.shape, dtype=bool)
                            roi_y_offset = y - y_start
                            roi_x_offset = x - x_start
                            surrounding_mask[roi_y_offset:roi_y_offset+h, 
                                           roi_x_offset:roi_x_offset+w] = False
                            
                            surrounding_pixels = surrounding[surrounding_mask]
                            if len(surrounding_pixels) > 0:
                                surrounding_mean = np.mean(surrounding_pixels)
                                contrast = abs(mean_intensity - surrounding_mean)
                                
                                # Accept defects with sufficient contrast
                                if contrast > 8:  # Lower threshold for subtle defects
                                    defects.append((x, y, w, h, 'gradient'))
                                    
                                    if self.debug_mode:
                                        print(f"  - Gradient defect: area={area}, contrast={contrast:.1f}, "
                                              f"mean_grad={mean_gradient:.1f}")
        
        if self.debug_mode:
            print(f"Total gradient defects found: {len(defects)}")
            
    except Exception as e:
        print(f"Error in gradient-based defect detection: {str(e)}")
        
    return defects

def _filter_and_merge_defects(self, defect_regions):
    """Enhanced filtering and merging of defects with better small defect preservation"""
    if not defect_regions:
        return []
    
    try:
        if self.debug_mode:
            print(f"\nFiltering {len(defect_regions)} detected defects...")
        
        # First, remove exact duplicates
        unique_defects = []
        seen = set()
        for defect in defect_regions:
            defect_tuple = tuple(defect)
            if defect_tuple not in seen:
                seen.add(defect_tuple)
                unique_defects.append(defect)
        
        if self.debug_mode:
            print(f"After removing duplicates: {len(unique_defects)} defects")
        
        # Sort by area (smaller defects first to preserve them)
        unique_defects.sort(key=lambda d: d[2] * d[3])  # Sort by area (w*h)
        
        # Filter overlapping defects with bias toward preserving smaller ones
        filtered_defects = []
        for i, defect in enumerate(unique_defects):
            x1, y1, w1, h1, type1 = defect
            area1 = w1 * h1
            
            # Check if this defect overlaps significantly with any already accepted defect
            overlaps = False
            for accepted_defect in filtered_defects:
                x2, y2, w2, h2, type2 = accepted_defect
                area2 = w2 * h2
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # Calculate overlap ratio relative to smaller defect
                min_area = min(area1, area2)
                overlap_ratio = overlap_area / min_area if min_area > 0 else 0
                
                # If overlap is significant, skip this defect
                if overlap_ratio > 0.5:  # 50% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                # Additional filtering based on defect characteristics
                if self._is_valid_defect(defect, unique_defects):
                    filtered_defects.append(defect)
        
        if self.debug_mode:
            print(f"After overlap filtering: {len(filtered_defects)} defects")
        
        return filtered_defects
        
    except Exception as e:
        print(f"Error in filtering defects: {str(e)}")
        return defect_regions

def _is_valid_defect(self, defect, all_defects):
    """Enhanced validation for defects, especially small ones"""
    x, y, w, h, defect_type = defect
    area = w * h
    
    # More lenient criteria for small defects
    if area < 100:  # Small defects
        # Accept if aspect ratio is reasonable
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio <= 8:  # Allow more elongated small defects
            return True
    
    # Standard criteria for larger defects
    elif area >= 100:
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio <= 5:  # Stricter for larger defects
            return True
    
    return False

def enhance_small_defect_detection(self, gray):
    """Additional method specifically for detecting very small defects"""
    defects = []
    try:
        if self.debug_mode:
            print("Running enhanced small defect detection...")
        
        # Use multiple scales for detection
        scales = [1.0, 0.8, 1.2]  # Original, slightly smaller, slightly larger
        
        for scale in scales:
            if scale != 1.0:
                # Resize image for multi-scale detection
                new_width = int(gray.shape[1] * scale)
                new_height = int(gray.shape[0] * scale)
                resized = cv2.resize(gray, (new_width, new_height))
            else:
                resized = gray
            
            # Apply bilateral filter to preserve edges while smoothing
            filtered = cv2.bilateralFilter(resized, 9, 75, 75)
            
            # Calculate difference between original and filtered
            diff = cv2.absdiff(resized, filtered)
            
            # Threshold the difference to find anomalies
            _, thresh = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
            
            # Clean up small noise
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 <= area <= 500:  # Focus on small to medium defects
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Scale coordinates back to original image size
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    # Validate the defect in original image
                    if (x + w < gray.shape[1] and y + h < gray.shape[0] and 
                        x >= 0 and y >= 0):
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            defects.append((x, y, w, h, 'small_defect'))
        
        # Remove duplicates from multi-scale detection
        if len(defects) > 1:
            defects = self._filter_duplicate_defects(defects, overlap_threshold=0.3)
        
        if self.debug_mode:
            print(f"Enhanced small defect detection found: {len(defects)} defects")
        
        return defects
        
    except Exception as e:
        print(f"Error in enhanced small defect detection: {str(e)}")
        return []


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
            
            # Edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Edge direction histogram
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Avoid division by zero
            grad_x = np.where(grad_x == 0, 1e-10, grad_x)
            angles = np.arctan2(grad_y, grad_x)
            
            # Quantize angles into 8 bins
            angles_quantized = np.digitize(angles, np.linspace(-np.pi, np.pi, 9)) - 1
            angle_hist = np.bincount(angles_quantized.flatten(), minlength=8)
            if angle_hist.sum() > 0:
                angle_hist = angle_hist / angle_hist.sum()
            features.extend(angle_hist)
            
        except Exception as e:
            print(f"Error in edge features: {str(e)}")
            features = [0.0] * 9  # Return zeros if calculation fails
            
        return features
    
    def _color_features(self, bgr_image, hsv_image):
        """Extract color-based features"""
        features = []
        try:
            # BGR channel statistics
            for channel in range(3):
                channel_data = bgr_image[:, :, channel]
                features.append(np.mean(channel_data))
                features.append(np.std(channel_data))
            
            # HSV channel statistics
            for channel in range(3):
                channel_data = hsv_image[:, :, channel]
                features.append(np.mean(channel_data))
                features.append(np.std(channel_data))
                
        except Exception as e:
            print(f"Error in color features: {str(e)}")
            features = [0.0] * 12  # Return zeros if calculation fails
            
        return features
    
    def _contour_features(self, gray):
        """Extract contour-based features"""
        features = []
        try:
            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Number of contours
                features.append(len(contours))
                
                # Statistics of contour areas
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                if areas:
                    features.append(np.mean(areas))
                    features.append(np.std(areas))
                    features.append(np.max(areas))
                    features.append(np.min(areas))
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                
                # Statistics of contour perimeters
                perimeters = [cv2.arcLength(c, True) for c in contours if cv2.arcLength(c, True) > 10]
                if perimeters:
                    features.append(np.mean(perimeters))
                    features.append(np.std(perimeters))
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0] * 7)
                
        except Exception as e:
            print(f"Error in contour features: {str(e)}")
            features = [0.0] * 7  # Return zeros if calculation fails
            
        return features
    
    def _frequency_features(self, gray):
        """Extract frequency domain features using FFT"""
        features = []
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Log transform to reduce dynamic range
            magnitude_spectrum = np.log(magnitude_spectrum + 1)
            
            # Extract features from frequency domain
            features.append(np.mean(magnitude_spectrum))
            features.append(np.std(magnitude_spectrum))
            features.append(np.max(magnitude_spectrum))
            features.append(np.min(magnitude_spectrum))
            
        except Exception as e:
            print(f"Error in frequency features: {str(e)}")
            features = [0.0] * 4  # Return zeros if calculation fails
            
        return features


# Fixed PCBDefectDetector class with more precise defect detection
class PCBDefectDetectorFixed:
    def __init__(self):
        # Much more conservative parameters to reduce false positives
        self.gaussian_kernel_size = 5
        self.min_defect_area = 15      # Increased significantly
        self.max_defect_area = 8000     # Limit maximum area
        self.bright_threshold = 240     # Much higher threshold
        self.dark_threshold = 25        # Much lower threshold
        self.edge_threshold_low = 150   # Higher edge thresholds
        self.edge_threshold_high = 250
        self.morphology_kernel_size = 7  # Larger kernel for better noise reduction
        
    def detect_and_highlight_defects(self, image):
        """Detect and highlight only significant defects with high precision"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply stronger noise reduction
            denoised = cv2.bilateralFilter(gray, 15, 35, 35)
            blurred = cv2.GaussianBlur(denoised, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
            
            # Create a copy of the original image for highlighting
            result_image = image.copy()
            
            # Only detect the most obvious defects
            defect_regions = []
            
            # Method 1: Detect extreme bright anomalies only
            extreme_bright_defects = self._detect_extreme_bright_defects(blurred, gray)
            defect_regions.extend(extreme_bright_defects)
            
            # Method 2: Detect missing components (large dark areas)
            missing_components = self._detect_missing_components(blurred, gray)
            defect_regions.extend(missing_components)
            
            # Method 3: Detect broken traces or major structural issues
            structural_breaks = self._detect_structural_breaks(blurred, gray)
            defect_regions.extend(structural_breaks)
            
            # Apply strict filtering
            filtered_defects = self._strict_defect_filtering(defect_regions, gray)
            
            # Highlight only the filtered defects
            defect_count = 0
            for defect in filtered_defects:
                x, y, w, h, defect_type, confidence = defect
                
                # Only show defects with high confidence
                if confidence > 0.7:
                    color_map = {
                        'extreme_bright': (0, 255, 255),    # Yellow for extreme bright
                        'missing_component': (255, 0, 0),   # Blue for missing components
                        'structural_break': (0, 0, 255),    # Red for structural breaks
                    }
                    
                    color = color_map.get(defect_type, (0, 0, 255))
                    
                    # Draw rectangle around defect
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
                    
                    # Draw circle at center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.circle(result_image, (center_x, center_y), 8, color, -1)
                    
                    # Add defect label with confidence
                    label = f"{defect_type.upper()} ({confidence:.2f})"
                    cv2.putText(result_image, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    defect_count += 1
            
            # Add defect count to image
            cv2.putText(result_image, f"Critical Defects: {defect_count}", 
                       (10, result_image.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return result_image
            
        except Exception as e:
            print(f"Error in defect detection: {str(e)}")
            return image

    def test_small_defect_detection(self, image_path, save_intermediate=False):
        """
        Test the small defect detection with detailed output
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load image")
                return None
            
            print(f"Testing small defect detection on: {image_path}")
            print(f"Image size: {image.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
            
            # Run all detection methods
            print("\nRunning all detection methods...")
            
            bright_defects = self._detect_bright_defects(blurred)
            print(f"Bright defects: {len(bright_defects)}")
            
            dark_defects = self._detect_dark_defects(blurred)
            print(f"Dark defects: {len(dark_defects)}")
            
            structural_defects = self._detect_structural_defects(blurred)
            print(f"Structural defects: {len(structural_defects)}")
            
            surface_defects = self._detect_surface_defects(blurred)
            print(f"Surface defects: {len(surface_defects)}")
            
            # Run enhanced small defect detection
            small_defects = self.enhance_small_defect_detection(blurred)
            print(f"Small defects: {len(small_defects)}")
            
            # Combine all defects
            all_defects = bright_defects + dark_defects + structural_defects + surface_defects + small_defects
            print(f"Total defects before filtering: {len(all_defects)}")
            
            # Filter defects
            filtered_defects = self._filter_and_merge_defects(all_defects)
            print(f"Final defects after filtering: {len(filtered_defects)}")
            
            # Create result image
            result_image = self.detect_and_highlight_defects(image)
            
            if save_intermediate:
                # Save intermediate results
                cv2.imwrite(f"debug_gray_{os.path.basename(image_path)}", gray)
                cv2.imwrite(f"debug_blurred_{os.path.basename(image_path)}", blurred)
            
            return result_image, filtered_defects
            
        except Exception as e:
            print(f"Error in test_small_defect_detection: {str(e)}")
            return None
    
    def _detect_extreme_bright_defects(self, blurred, original_gray):
        """Detect only extreme bright defects like severe burn marks or excess solder"""
        defects = []
        try:
            # Use adaptive thresholding to account for lighting variations
            mean_intensity = np.mean(blurred)
            adaptive_threshold = max(self.bright_threshold, mean_intensity + 50)
            
            # Create mask for extremely bright areas
            _, bright_mask = cv2.threshold(blurred, adaptive_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply strong morphological operations to reduce noise
            kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_defect_area:
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    
                    # Calculate confidence based on intensity contrast
                    roi = original_gray[y:y+h, x:x+w]
                    surrounding_area = self._get_surrounding_area(original_gray, x, y, w, h)
                    
                    if surrounding_area is not None and len(surrounding_area) > 0:
                        contrast = np.mean(roi) - np.mean(surrounding_area)
                        confidence = min(1.0, contrast / 100.0)  # Normalize confidence
                        
                        if confidence > 0.5:  # Only high confidence defects
                            defects.append((x, y, w, h, 'extreme_bright', confidence))
                            
        except Exception as e:
            print(f"Error detecting extreme bright defects: {str(e)}")
            
        return defects
    
    def _detect_missing_components(self, blurred, original_gray):
        """Detect missing components (large dark areas where components should be)"""
        defects = []
        try:
            # Use adaptive thresholding for dark areas
            mean_intensity = np.mean(blurred)
            adaptive_threshold = min(self.dark_threshold, mean_intensity - 30)
            
            # Create mask for extremely dark areas
            _, dark_mask = cv2.threshold(blurred, adaptive_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations
            kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_defect_area:
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    
                    # Check if it's really a missing component by analyzing surroundings
                    roi = original_gray[y:y+h, x:x+w]
                    surrounding_area = self._get_surrounding_area(original_gray, x, y, w, h)
                    
                    if surrounding_area is not None and len(surrounding_area) > 0:
                        # Missing components should have bright surroundings
                        contrast = np.mean(surrounding_area) - np.mean(roi)
                        confidence = min(1.0, contrast / 80.0)
                        
                        # Also check if the dark area is roughly rectangular (component-like)
                        aspect_ratio = w / h
                        if 0.5 <= aspect_ratio <= 2.0 and confidence > 0.6:
                            defects.append((x, y, w, h, 'missing_component', confidence))
                            
        except Exception as e:
            print(f"Error detecting missing components: {str(e)}")
            
        return defects
    
    def _detect_structural_breaks(self, blurred, original_gray):
        """Detect structural breaks in traces or major damage"""
        defects = []
        try:
            # Use Canny edge detection with higher thresholds
            edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
            
            # Look for breaks in continuity (gaps in edges where there should be connections)
            # This is a simplified approach - in practice, you'd need template matching
            
            # Apply morphological operations to find disconnected regions
            kernel = np.ones((15, 15), np.uint8)  # Larger kernel to find major breaks
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Find contours of potential breaks
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_defect_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's a real structural break
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Calculate form factor (4π*area/perimeter²)
                        form_factor = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Structural breaks usually have irregular shapes
                        if form_factor < 0.5:  # Irregular shape
                            confidence = 1.0 - form_factor  # More irregular = higher confidence
                            defects.append((x, y, w, h, 'structural_break', confidence))
                            
        except Exception as e:
            print(f"Error detecting structural breaks: {str(e)}")
            
        return defects
    
    def _get_surrounding_area(self, image, x, y, w, h):
        """Get pixels surrounding a region for contrast analysis"""
        try:
            # Define surrounding area (border around the region)
            border_size = 20
            y_start = max(0, y - border_size)
            y_end = min(image.shape[0], y + h + border_size)
            x_start = max(0, x - border_size)
            x_end = min(image.shape[1], x + w + border_size)
            
            # Extract surrounding area
            surrounding_roi = image[y_start:y_end, x_start:x_end]
            
            # Create mask to exclude the central region
            mask = np.ones(surrounding_roi.shape, dtype=bool)
            inner_y_start = max(0, y - y_start)
            inner_y_end = min(surrounding_roi.shape[0], y + h - y_start)
            inner_x_start = max(0, x - x_start)
            inner_x_end = min(surrounding_roi.shape[1], x + w - x_start)
            
            mask[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = False
            
            return surrounding_roi[mask]
            
        except Exception as e:
            print(f"Error getting surrounding area: {str(e)}")
            return None
    
    def _strict_defect_filtering(self, defects, gray):
        """Apply strict filtering to remove false positives"""
        filtered_defects = []
        
        for defect in defects:
            x, y, w, h, defect_type, confidence = defect
            
            # Additional checks for each defect type
            if defect_type == 'extreme_bright':
                # Check if it's really an anomaly and not just a shiny component
                roi = gray[y:y+h, x:x+w]
                intensity_uniformity = np.std(roi)
                if intensity_uniformity < 20 and confidence > 0.6:  # Uniform bright area
                    filtered_defects.append(defect)
                    
            elif defect_type == 'missing_component':
                # Check if the dark area is really where a component should be
                roi = gray[y:y+h, x:x+w]
                # Missing components should be relatively uniform dark areas
                intensity_uniformity = np.std(roi)
                if intensity_uniformity < 15 and confidence > 0.7:
                    filtered_defects.append(defect)
                    
            elif defect_type == 'structural_break':
                # Only keep high-confidence structural breaks
                if confidence > 0.8:
                    filtered_defects.append(defect)
        
        return filtered_defects


if __name__ == "__main__":
    # Create the inspection system with enhanced defect detector
    inspector = PCBInspectionSystem()
    
    # Initialize the enhanced defect detector with debug mode
    inspector.defect_detector = PCBDefectDetectorFixed()
    inspector.defect_detector.debug_mode = True  # Enable debug output
    
    # Set parameters optimized for small defect detection
    inspector.defect_detector.min_defect_area = 15  # Lower threshold for small defects
    inspector.defect_detector.bright_threshold = 180  # Adjust based on your images
    inspector.defect_detector.dark_threshold = 80    # Adjust based on your images
    
    # Check if model exists, if not train it
    if not inspector.load_model():
        print("No trained model found. Training new model...")
        success = inspector.train_model()
        if not success:
            print("Training failed. Please check your dataset.")
            exit()
    
    # Interactive menu
    while True:
        print("\n=== Enhanced PCB Inspection System ===")
        print("1. Train new model")
        print("2. Inspect single PCB image")
        print("3. Batch inspect directory")
        print("4. View training statistics")
        print("5. Test defect detection only (no classification)")
        print("6. Configure detection parameters")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            good_dir = input("Enter path to good PCB images directory (default: dataset/good): ").strip()
            if not good_dir:
                good_dir = "dataset/good"
            
            defective_dir = input("Enter path to defective PCB images directory (default: dataset/defective): ").strip()
            if not defective_dir:
                defective_dir = "dataset/defective"
            
            success = inspector.train_model(good_dir, defective_dir)
            if success:
                print("Model trained successfully!")
            else:
                print("Training failed. Please check your dataset.")
        
        elif choice == '2':
            image_path = input("Enter path to PCB image: ").strip()
            if os.path.exists(image_path):
                result = inspector.inspect_pcb(image_path)
                if result:
                    print(f"\nInspection Result:")
                    print(f"Image: {result['image_path']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print(f"Timestamp: {result['timestamp']}")
                else:
                    print("Failed to inspect the image.")
            else:
                print("Image file not found.")
        
        elif choice == '3':
            test_dir = input("Enter path to test images directory (default: test_images): ").strip()
            if not test_dir:
                test_dir = "test_images"
            
            results = inspector.batch_inspect(test_dir)
            if results:
                print(f"\nBatch Inspection Results ({len(results)} images):")
                good_count = sum(1 for r in results if r['prediction'] == 'Good')
                defective_count = len(results) - good_count
                print(f"Good PCBs: {good_count}")
                print(f"Defective PCBs: {defective_count}")
            else:
                print("No results to display.")
        
        elif choice == '4':
            if os.path.exists(inspector.stats_path):
                with open(inspector.stats_path, 'r') as f:
                    stats = json.load(f)
                print(f"\nTraining Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
            else:
                print("No training statistics found. Train a model first.")
        
        elif choice == '5':
            # NEW: Test defect detection only (useful for debugging)
            image_path = input("Enter path to PCB image for defect detection: ").strip()
            if os.path.exists(image_path):
                try:
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is not None:
                        print(f"\nProcessing image: {image_path}")
                        print("Running enhanced defect detection...")
                        
                        # Run defect detection with enhanced methods
                        result_image = inspector.defect_detector.detect_and_highlight_defects(image)
                        
                        # Save result
                        output_path = f"detected_defects_{os.path.basename(image_path)}"
                        cv2.imwrite(output_path, result_image)
                        print(f"Result saved as: {output_path}")
                        
                        # Display using matplotlib if available
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(15, 5))
                            
                            plt.subplot(1, 2, 1)
                            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            plt.title('Original Image')
                            plt.axis('off')
                            
                            plt.subplot(1, 2, 2)
                            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                            plt.title('Detected Defects')
                            plt.axis('off')
                            
                            plt.tight_layout()
                            plt.show()
                        except ImportError:
                            print("Matplotlib not available. Result saved to file only.")
                    else:
                        print("Failed to load image.")
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
            else:
                print("Image file not found.")
        
        elif choice == '6':
            # NEW: Configure detection parameters
            print("\nCurrent Detection Parameters:")
            print(f"Min defect area: {inspector.defect_detector.min_defect_area}")
            print(f"Bright threshold: {inspector.defect_detector.bright_threshold}")
            print(f"Dark threshold: {inspector.defect_detector.dark_threshold}")
            print(f"Debug mode: {inspector.defect_detector.debug_mode}")
            
            print("\nAdjust parameters (press Enter to keep current value):")
            
            try:
                new_min_area = input(f"Min defect area ({inspector.defect_detector.min_defect_area}): ").strip()
                if new_min_area:
                    inspector.defect_detector.min_defect_area = int(new_min_area)
                
                new_bright_thresh = input(f"Bright threshold ({inspector.defect_detector.bright_threshold}): ").strip()
                if new_bright_thresh:
                    inspector.defect_detector.bright_threshold = int(new_bright_thresh)
                
                new_dark_thresh = input(f"Dark threshold ({inspector.defect_detector.dark_threshold}): ").strip()
                if new_dark_thresh:
                    inspector.defect_detector.dark_threshold = int(new_dark_thresh)
                
                debug_choice = input(f"Debug mode (y/n, current: {'y' if inspector.defect_detector.debug_mode else 'n'}): ").strip().lower()
                if debug_choice in ['y', 'n']:
                    inspector.defect_detector.debug_mode = (debug_choice == 'y')
                
                print("Parameters updated successfully!")
            except ValueError:
                print("Invalid input. Parameters not changed.")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-7.")

print("Script completed...")