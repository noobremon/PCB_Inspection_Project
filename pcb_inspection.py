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


class PCBDefectDetector:
    def __init__(self):
        self.gaussian_kernel_size = 5
        self.threshold_value = 50
        self.min_contour_area = 100
        
    def detect_and_highlight_defects(self, image):
        """Detect potential defects and highlight them with circles"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Apply morphological operations to enhance defects
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a copy of the original image for highlighting
            result_image = image.copy()
            
            # Additional defect detection methods
            defect_regions = []
            
            # Method 1: Detect bright spots (potential soldering defects)
            bright_spots = self._detect_bright_spots(gray)
            defect_regions.extend(bright_spots)
            
            # Method 2: Detect dark spots (potential missing components)
            dark_spots = self._detect_dark_spots(gray)
            defect_regions.extend(dark_spots)
            
            # Method 3: Detect irregular contours
            irregular_contours = self._detect_irregular_contours(contours)
            defect_regions.extend(irregular_contours)
            
            # Method 4: Detect texture anomalies
            texture_anomalies = self._detect_texture_anomalies(gray)
            defect_regions.extend(texture_anomalies)
            
            # Highlight all detected defect regions
            for region in defect_regions:
                x, y, w, h = region
                # Draw rectangle around defect
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Draw circle at center
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(result_image, (center_x, center_y), max(w, h) // 2, (0, 255, 255), 2)
                # Add defect label
                cv2.putText(result_image, "DEFECT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # If no specific defects found, highlight areas with high edge density
            if not defect_regions:
                result_image = self._highlight_edge_dense_areas(result_image, edges)
            
            return result_image
            
        except Exception as e:
            print(f"Error in defect detection: {str(e)}")
            return image
    
    def _detect_bright_spots(self, gray):
        """Detect bright spots that might indicate soldering defects"""
        spots = []
        try:
            # Threshold for bright spots
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours of bright spots
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    spots.append((x, y, w, h))
                    
        except Exception as e:
            print(f"Error detecting bright spots: {str(e)}")
            
        return spots
    
    def _detect_dark_spots(self, gray):
        """Detect dark spots that might indicate missing components"""
        spots = []
        try:
            # Threshold for dark spots
            _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of dark spots
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    spots.append((x, y, w, h))
                    
        except Exception as e:
            print(f"Error detecting dark spots: {str(e)}")
            
        return spots
    
    def _detect_irregular_contours(self, contours):
        """Detect irregular contours that might indicate defects"""
        irregular_regions = []
        try:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Calculate contour properties
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Circularity measure
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # If contour is very irregular (low circularity), mark as potential defect
                        if circularity < 0.3:
                            x, y, w, h = cv2.boundingRect(contour)
                            irregular_regions.append((x, y, w, h))
                            
        except Exception as e:
            print(f"Error detecting irregular contours: {str(e)}")
            
        return irregular_regions
    
    def _detect_texture_anomalies(self, gray):
        """Detect texture anomalies using local standard deviation"""
        anomalies = []
        try:
            # Calculate local standard deviation
            kernel = np.ones((15, 15), np.float32) / 225
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            std_dev = np.sqrt(sqr_mean - mean**2)
            
            # Threshold for high texture variation
            _, texture_mask = cv2.threshold(std_dev.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
            
            # Find contours of texture anomalies
            contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filter small variations
                    x, y, w, h = cv2.boundingRect(contour)
                    anomalies.append((x, y, w, h))
                    
        except Exception as e:
            print(f"Error detecting texture anomalies: {str(e)}")
            
        return anomalies
    
    def _highlight_edge_dense_areas(self, image, edges):
        """Highlight areas with high edge density as potential defects"""
        try:
            # Divide image into regions and calculate edge density
            height, width = edges.shape
            region_size = 50
            
            for y in range(0, height - region_size, region_size):
                for x in range(0, width - region_size, region_size):
                    region = edges[y:y+region_size, x:x+region_size]
                    edge_density = np.sum(region) / (region_size * region_size * 255)
                    
                    # If edge density is high, mark as potential defect area
                    if edge_density > 0.1:
                        cv2.rectangle(image, (x, y), (x + region_size, y + region_size), (255, 0, 0), 1)
                        cv2.putText(image, "ANOMALY", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                        
        except Exception as e:
            print(f"Error highlighting edge dense areas: {str(e)}")
            
        return image


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
        
        if result['prediction'] == 'Defective':
            print("Defect visualization has been displayed in a popup window.")
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
        
        if defective_count > 0:
            print("Defective images were displayed with visual analysis.")

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