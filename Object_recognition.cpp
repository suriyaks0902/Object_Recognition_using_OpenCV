/*
  Suriya Kasiyalan Siva
  02/24/2024
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <fstream>
#include <sys/stat.h>

using namespace cv;
using namespace std;

class Object;
class FeatureStdDeviations;

// Declarations
static map<string, double> compute_features(const Mat& regionMask, Mat& frame);
static void saveObjectData(const string& label, const map<string, double>& features, ofstream& outputFile);
static vector<vector<int>> buildConfusionMatrix(const vector<string>& true_labels, const vector<string>& predicted_labels);
static void printConfusionMatrix(const vector<vector<int>>& confusion_matrix);

// Confusion Matrix and Number of Classes
const int max_img_class = 100;
const int num_classes = 5;
vector<pair<string, string>> userFeedbacks; // Stores user feedbacks
cv::Mat confusionMatrix = cv::Mat::zeros(num_classes, num_classes, CV_32S); // Initialize confusion matrix



static vector<vector<int>> buildConfusionMatrix(const vector<string>& true_labels, const vector<string>& predicted_labels) {

    unordered_map<string, vector<int>> confusion_map;


    for (size_t i = 0; i < true_labels.size(); ++i) {
        const string& true_label = true_labels[i];
        const string& predicted_label = predicted_labels[i];


        if (confusion_map.find(true_label) == confusion_map.end()) {
            confusion_map[true_label] = { 0, 0, 0 };
        }
        if (confusion_map.find(predicted_label) == confusion_map.end()) {
            confusion_map[predicted_label] = { 0, 0, 0 };
        }


        if (true_label == predicted_label) {
            confusion_map[true_label][0]++;
        }
        else {
            confusion_map[predicted_label][1]++;
            confusion_map[true_label][2]++;
        }
    }


    vector<vector<int>> confusion_matrix;
    for (const auto& pair : confusion_map) {
        confusion_matrix.push_back(pair.second);
    }

    return confusion_matrix;
}


static void printConfusionMatrix(const vector<vector<int>>& confusion_matrix) {
    cout << "Confusion Matrix:" << endl;
    for (const auto& row : confusion_matrix) {
        for (int cell : row) {
            cout << cell << "\t";
        }
        cout << endl;
    }
}


static map<string, double> compute_features(const Mat& regionMask, Mat& frame);
class Object {
public:
    double centroid_x;
    double centroid_y;
    double theta;
    double percent_filled;
    double bounding_box_ratio;
    string label;

    Object(double x, double y, double t, double pf, double bbr, const string& l) : centroid_x(x), centroid_y(y), theta(t), percent_filled(pf), bounding_box_ratio(bbr), label(l) {}
};

class FeatureStdDeviations {
public:
    double centroid_x_stddev;
    double centroid_y_stddev;
    double theta_stddev;
    double percent_filled_stddev;
    double bounding_box_ratio_stddev;
};

// Function to check if a file exists
bool fileExists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

static double computeStdDev(const vector<double>& values);
static FeatureStdDeviations computeFeatureStdDeviations(const vector<Object>& known_objects) {
    FeatureStdDeviations stddev;
    int num_objects = known_objects.size();
    if (num_objects == 0) {
        throw std::invalid_argument("No known objects provided.");
    }

    vector<double> centroid_x_values, centroid_y_values, theta_values, percent_filled_values, bounding_box_ratio_values;

    for (const auto& obj : known_objects) {
        centroid_x_values.push_back(obj.centroid_x);
        centroid_y_values.push_back(obj.centroid_y);
        theta_values.push_back(obj.theta);
        percent_filled_values.push_back(obj.percent_filled);
        bounding_box_ratio_values.push_back(obj.bounding_box_ratio);
    }

    stddev.centroid_x_stddev = computeStdDev(centroid_x_values);
    stddev.centroid_y_stddev = computeStdDev(centroid_y_values);
    stddev.theta_stddev = computeStdDev(theta_values);
    stddev.percent_filled_stddev = computeStdDev(percent_filled_values);
    stddev.bounding_box_ratio_stddev = computeStdDev(bounding_box_ratio_values);

    return stddev;
}

static double computeStdDev(const vector<double>& values) {
    double mean = 0.0;
    for (double value : values) {
        mean += value;
    }
    mean /= values.size();

    double variance = 0.0;
    for (double value : values) {
        variance += pow(value - mean, 2);
    }
    variance /= values.size();

    return sqrt(variance);
}

static double scaledEuclideanDistance(const Object& f1, const Object& f2, const FeatureStdDeviations& stddev) {
    double scaled_percent_filled_diff = (f1.percent_filled - f2.percent_filled) / stddev.percent_filled_stddev;
    double scaled_bounding_box_ratio_diff = (f1.bounding_box_ratio - f2.bounding_box_ratio) / stddev.bounding_box_ratio_stddev;

    double scaled_distance = sqrt(pow(scaled_percent_filled_diff, 2) + pow(scaled_bounding_box_ratio_diff, 2));
    return scaled_distance;
}

static string classifyObject(const Object& unknown_object, const vector<Object>& known_objects, const FeatureStdDeviations& stddev) {
    double min_distance = numeric_limits<double>::infinity();
    string closest_object_label;

    for (const auto& known_object : known_objects) {
        double distance = scaledEuclideanDistance(unknown_object, known_object, stddev);
        if (distance < min_distance) {
            min_distance = distance;
            closest_object_label = known_object.label;
        }
    }

    return closest_object_label;
}

static int dynmcThresholding(const Mat& image) {
    Mat samples = image.reshape(1, image.rows * image.cols);
    samples.convertTo(samples, CV_32F);
    int K = 2;
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.2);
    kmeans(samples, K, labels, criteria, 10, KMEANS_RANDOM_CENTERS, centers);
    double threshold = (centers.at<float>(0) + centers.at<float>(1)) / 2.0;
    return static_cast<int>(threshold);
}

static void normalThresholding(Mat& image, int thresholdvalue, Mat& result) {
    Mat gray_img;
    if (image.channels() == 3)
        cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
    else
        gray_img = image.clone();

    result = Mat(gray_img.rows, gray_img.cols, CV_8UC1);

    for (int y = 0; y < gray_img.rows; ++y) {
        for (int x = 0; x < gray_img.cols; ++x) {
            if (gray_img.at<uchar>(y, x) >= thresholdvalue)
                result.at<uchar>(y, x) = 0;
            else
                result.at<uchar>(y, x) = 255;
        }
    }
}

map<string, double> compute_features(const Mat& regionMask, Mat& frame) {
    map<string, double> features;

    // Check if the region mask contains any nonzero pixels
    if (countNonZero(regionMask) == 0) {
        return features; // No object present, return empty features
    }
    // Compute moments
    Moments moments = cv::moments(regionMask, false); // Use binary moments

    // Calculate area
    double area = moments.m00;

    // Calculate centroid
    double xc = moments.m10 / area;
    double yc = moments.m01 / area;

    // Calculate central moments
    double mu20 = moments.mu20 / area;
    double mu02 = moments.mu02 / area;
    double mu11 = moments.mu11 / area;

    // Calculate angle of least central moment
    double alpha = 0.5 * atan2(2 * mu11, (mu20 - mu02));

    // Calculate axis of the least central moment from centroid
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);
    Point2f start(xc, yc);
    Point2f end(xc + 50 * cos_alpha, yc + 50 * sin_alpha); // Adjust length of axis pointer as needed
    line(frame, start, end, Scalar(0, 0, 255), 2);

    // Calculate rotated bounding box points
    vector<Point> nonzeroPoints;
    findNonZero(regionMask, nonzeroPoints);
    RotatedRect rotatedRect = minAreaRect(nonzeroPoints);

    // Extract width and height from the rotated rectangle
    double width = rotatedRect.size.width;
    double height = rotatedRect.size.height;

    // Draw oriented bounding box
    Point2f rectPoints[4];
    rotatedRect.points(rectPoints);
    for (int i = 0; i < 4; ++i) {
        line(frame, rectPoints[i], rectPoints[(i + 1) % 4], Scalar(255, 0, 255), 2);
    }

    // Calculate percent filled
    double bounding_box_area = width * height;
    double percent_filled = (area / bounding_box_area) * 100;

    // Store computed features in the map
    features["centroid_x"] = xc;
    features["centroid_y"] = yc;
    features["theta"] = alpha;
    features["percent_filled"] = percent_filled;
    features["bounding_box_ratio"] = height / width;

    return features;
}

void saveObjectData(const string& label, const map<string, double>& features, ofstream& outputFile) {
    if (outputFile.is_open()) {
        outputFile << label << ",";
        outputFile << features.at("centroid_x") << ",";
        outputFile << features.at("centroid_y") << ",";
        outputFile << features.at("theta") << ",";
        outputFile << features.at("percent_filled") << ",";
        outputFile << features.at("bounding_box_ratio") << endl;

        cout << "Object data saved with label: " << label << endl;
    }
    else {
        cerr << "Error: Unable to write to the output file. File is not open." << endl;
    }
}

class KNNClassifier {
private:
    int K;
    vector<Object> known_objects;
    FeatureStdDeviations stddev;

    // Calculate the distance between two objects
    double distance(const Object& obj1, const Object& obj2) {
        // Implement your distance metric here
        double distance = sqrt(pow(obj1.centroid_x - obj2.centroid_x, 2) + pow(obj1.centroid_y - obj2.centroid_y, 2));
        return distance;
    }

public:
    // Constructor
    KNNClassifier(int k, const vector<Object>& objects, const FeatureStdDeviations& std_dev) : K(k), known_objects(objects), stddev(std_dev) {}

    // Classify a new object
    string classify(const Object& unknown_object) {
        vector<pair<double, string>> distances_and_labels;

        // Calculate distances from unknown object to known objects
        for (const auto& known_object : known_objects) {
            double dist = distance(unknown_object, known_object);
            distances_and_labels.emplace_back(dist, known_object.label);
        }

        // Sort distances in ascending order
        sort(distances_and_labels.begin(), distances_and_labels.end());

        // Count the occurrences of each label among the K nearest neighbors
        unordered_map<string, int> label_counts;
        for (int i = 0; i < K; ++i) {
            label_counts[distances_and_labels[i].second]++;
        }

        // Find the majority class among the K nearest neighbors
        string majority_class;
        int max_count = numeric_limits<int>::min();
        for (const auto& pair : label_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                majority_class = pair.first;
            }
        }

        return majority_class;
    }
};

int main() {

    // Set threshold distance for classification
    double threshold_distance = 100.0; // Adjust threshold distance as needed

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the webcam.\n";
        return -1;
    }

    ofstream outputFile("object_data.csv", ios::app);
    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open object_data.csv for writing.\n";
        return -1;
    }

    // Check if the file exists, if not, write the header
    if (!fileExists("object_data.csv")) {
        outputFile << "Label,Centroid_X,Centroid_Y,Theta,Percent_Filled,BoundingBox_Ratio\n";
    }
    
    
    char mode = 'n'; // Mode: 'n' for normal operation, 't' for training mode
    string label; // Label for the object

    vector<Object> known_objects;
    known_objects.clear();
    ifstream objectFile("object_data.csv");
    if (objectFile.is_open()) {
        string line;
        getline(objectFile, line); // Skip header line
        while (getline(objectFile, line)) {
            stringstream ss(line);
            string label;
            double centroid_x, centroid_y, theta, percent_filled, bounding_box_ratio;
            char delimiter;
            if (getline(ss, label, ',') && (ss >> centroid_x >> delimiter >> centroid_y >> delimiter >> theta >> delimiter >> percent_filled >> delimiter >> bounding_box_ratio)) {
                known_objects.push_back(Object(centroid_x, centroid_y, theta, percent_filled, bounding_box_ratio, label));
            }
        }
        objectFile.close();
    }
    else {
        cerr << "Error: Unable to open object_data.csv for reading.\n";
        return -1;
    }

    FeatureStdDeviations stddev = computeFeatureStdDeviations(known_objects);
    int k =20;

    // Initialize KNN classifier
    KNNClassifier knn_classifier(k, known_objects, stddev);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Unable to capture frame.\n";
            break;
        }
    
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    
        int thresholdValue = dynmcThresholding(grayFrame);
        Mat thresholded;
        normalThresholding(grayFrame, thresholdValue, thresholded);
    
        Mat erodedImage;
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        erode(thresholded, erodedImage, element);
    
        Mat labels, stats, centroids;
        int numComponents = connectedComponentsWithStats(erodedImage, labels, stats, centroids, 8, CV_32S);
    
        int minRegionSize = 100;
        int maxRegionsToShow = 1;
        vector<int> regionIndices;
        for (int j = 1; j < numComponents; ++j) {
            if (stats.at<int>(j, CC_STAT_AREA) > minRegionSize) {
                regionIndices.push_back(j);
            }
        }
        sort(regionIndices.begin(), regionIndices.end(),
            [&](int a, int b) { return stats.at<int>(a, CC_STAT_AREA) > stats.at<int>(b, CC_STAT_AREA); });
    
        Mat regionMap = Mat::zeros(labels.size(), CV_8UC3);
        for (int index : regionIndices) {
            Mat regionMask = (labels == index);
            map<string, double> features = compute_features(regionMask, frame);
            Object unknown_object(features["centroid_x"], features["centroid_y"], features["theta"],
                features["percent_filled"], features["bounding_box_ratio"], "");
            string object_label = classifyObject(unknown_object, known_objects, stddev);
    
            // Display the region and ask for the true label
    
    
            // Display the label on the frame
            if (!object_label.empty()) {
                cout << "Object label: " << object_label << endl; // Debug statement
                cout << "Centroid position: (" << features["centroid_x"] << ", " << features["centroid_y"] << ")" << endl; // Debug statement
                putText(frame, object_label, Point(features["centroid_x"], features["centroid_y"]),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    
            }
            else {
                cout << "Object label is empty!" << endl; // Debug statement
            }
            regionMap.setTo(Scalar(rand() & 255, rand() & 255, rand() & 255), regionMask);
            if (mode == 't' && label.empty()) { // Training mode and label is not set
                cout << "Enter label for the object: ";
                cin >> label;
            }
            // If label is provided, save object data and store in known_objects vector
            if (!label.empty()) {
                saveObjectData(label, features, outputFile);
                known_objects.emplace_back(features["centroid_x"], features["centroid_y"], features["theta"],
                    features["percent_filled"], features["bounding_box_ratio"], label);
            }
            label.clear();
            cout << "Region " << index << " Features:\n";
            for (const auto& pair : features) {
                cout << pair.first << ": " << pair.second << endl;
            }
            regionMap.setTo(Scalar(rand() & 255, rand() & 255, rand() & 255), regionMask);
        }
    
        // Display various images
        //imshow("Thresholded Image", thresholded);
        //imshow("Cleaned Up Binary Image", erodedImage);
        imshow("Frame", frame);
        imshow("Region Map", regionMap);
    
        char key = waitKey(30); 
        cout << "Pressed key: " << key << endl;
        if (key == 't' || key == 'T') { // Toggle training mode on 'T' key press
            mode = (mode == 't') ? 'n' : 't'; // Switch mode
            if (mode == 't') {
                cout << "Training mode enabled. Press 'T' again to disable.\n";
            }
            else {
                cout << "Training mode disabled.\n";
                label.clear(); // Clear label when training mode is disabled
            }
        }
        
        if (key == 'c' || key == 'C') {
            // Enter confusion mode
            cout << "Entering confusion mode..." << endl;
            string trueLabel;
            cout << "Enter the true label for the object: ";
            cin >> trueLabel;
            for (int index : regionIndices) {
                Mat regionMask = (labels == index);
                map<string, double> features = compute_features(regionMask, frame);
                // Perform necessary operations for each region
                Object unknown_object = Object(features["centroid_x"], features["centroid_y"], features["theta"],
                    features["percent_filled"], features["bounding_box_ratio"], "");
    
    
                // Classify the object
                string classifiedLabel = classifyObject(unknown_object, known_objects, stddev);
    
                // Get user feedback
                bool correctClassification = (trueLabel == classifiedLabel);
    
                // Update user feedbacks
                userFeedbacks.push_back({ trueLabel, classifiedLabel });
            }
    
            // Build confusion matrix
            vector<string> trueLabels, classifiedLabels;
            for (const auto& feedback : userFeedbacks) {
                trueLabels.push_back(feedback.first);
                classifiedLabels.push_back(feedback.second);
            }
            vector<vector<int>> confusionMatrix = buildConfusionMatrix(trueLabels, classifiedLabels);
            // Print confusion matrix
            printConfusionMatrix(confusionMatrix);
    
        }
    
        
        else if (key == 27) { 
            cout << "Esc key is pressed. Exiting...\n";
            break;
        }
    }
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Unable to capture frame.\n";
            break;
        }
        
        // Perform object detection
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        int thresholdValue = dynmcThresholding(grayFrame);
        Mat thresholded;
        normalThresholding(grayFrame, thresholdValue, thresholded);

        Mat erodedImage;
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        erode(thresholded, erodedImage, element);

        Mat labels, stats, centroids;
        int numComponents = connectedComponentsWithStats(erodedImage, labels, stats, centroids, 8, CV_32S);

        int minRegionSize = 100;
        vector<int> regionIndices;
        for (int j = 1; j < numComponents; ++j) {
            if (stats.at<int>(j, CC_STAT_AREA) > minRegionSize) {
                regionIndices.push_back(j);
            }
        }

        // Extract features and classify each detected object
        for (int index : regionIndices) {
            Mat regionMask = (labels == index);
            map<string, double> features = compute_features(regionMask, frame);
            Object unknown_object(features["centroid_x"], features["centroid_y"], features["theta"],
                features["percent_filled"], features["bounding_box_ratio"], "");

            // Classify the object
            string predicted_label = knn_classifier.classify(unknown_object);
            putText(frame, predicted_label, Point(features["centroid_x"], features["centroid_y"]),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
            
        }
        cout << "Entered KNN Classifier loop...\n";
        // Display the frame
        imshow("KNN Classifier frame", frame);
        char key = waitKey(30);
        if (key == 27) {
            cout << "Esc key is pressed. Exiting...\n";
            break;
        }
    }
    cap.release();
    outputFile.close(); // Close the output file when done

    return 0;

}
