# Real-time 2-D Object Recognition System

## Overview
This project implements a real-time 2-D object recognition system using C++ and OpenCV. The system is designed to identify specific objects placed on a white surface, regardless of their translation, scale, or rotation. It leverages computer vision techniques to process input images or video streams from a downward-facing camera.

## Key Features
- **Thresholding and Preprocessing:** Utilizes dynamic thresholding algorithms and Gaussian blur for noise reduction and refinement of image details.
- **Morphological Filtering:** Cleans up binary images using morphological filtering techniques, specifically the shrinking approach with erosion to remove noise and small protrusions.
- **Image Segmentation:** Implements connected components analysis for segmenting input images into distinct regions, filtering out small regions and visualizing segmented regions with a fixed color palette.
- **Feature Extraction:** Computes essential features for each major region, such as centroid position, orientation, and bounding box ratio, and visualizes them on the original frame.
- **Training Data Collection:** Implements a training mode to collect feature vectors from known objects and store them in a CSV file, allowing users to assign labels during the training process.
- **Object Classification:** Develops object recognition system capable of classifying objects in real-time video streams using nearest-neighbor and K-Nearest Neighbor (KNN) classification approaches.
- **User Feedback and Evaluation:** Enables users to provide feedback on object classification accuracy, allowing for continuous evaluation and improvement of the system.
- **Confusion Matrix Analysis:** Evaluates system performance through confusion matrix analysis, identifying misclassifications to improve accuracy.
- **Extensions:** Expands the object database to recognize more objects and enables automatic learning of new objects, demonstrating the system's versatility and adaptability.

## Video Demonstration
To provide a visual overview of the Real-time 2-D Object Recognition System in action, check out the following video demonstration:


https://github.com/suriyaks0902/Real_Time_2D_Object_Recognition_using_OpenCV/assets/90970879/1ac2376d-3703-4290-97f9-39055f904171




https://github.com/suriyaks0902/Real_Time_2D_Object_Recognition_using_OpenCV/assets/90970879/ab414def-a740-48d4-ab76-657b178eb8b0




## Installation
1. **Prerequisites:**
   - Visual Studio (or any C++ compiler)
   - OpenCV library for C++
2. **Clone the Repository:**
3. **Install Dependencies:**
   - Set up OpenCV library for C++ in your Visual Studio project.
4. **Build and Run:**
   - Open the solution file in Visual Studio.
   - Build the project.
   - Connect a webcam to your computer.
   - Run the executable to start the object recognition system.

## Usage
1. **Object Training Mode:**
   - Press 'T' to enter object training mode.
   - Place the object of interest in front of the camera.
   - Enter the label for the object when prompted.
   - Repeat the process for each object you want to train.
   - Press 'T' again to exit training mode.
2. **Normal Operation:**
   - The system will detect and classify objects in real-time once trained.
   - Objects will be labeled and displayed on the video stream.
   - Press 'C' to enter confusion mode and provide feedback on object classification accuracy.
3. **Exiting the Program:**
   - Press 'Esc' key to exit the program.

## Contributing
Contributions to the project are welcome! Feel free to submit bug fixes, enhancements, or new features via pull requests.

## Acknowledgments
We acknowledge the invaluable support and guidance received during the development of this project.

## Contact
For any inquiries or feedback, please contact [k.s.suriya0902@gmail.com].

Thank you for your interest in the Real-time 2-D Object Recognition System! We hope you find it useful and engaging.
