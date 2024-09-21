
# Classification Methods for Lung Cancer Detection

This project demonstrates the implementation of various classification methods to predict the level of lung cancer in patients using different machine learning models. The models used in this project include Support Vector Machines (SVM), Decision Trees, Random Forest, and a Neural Network.

## Requirements

To run the scripts provided, ensure you have the following libraries installed:

### Common Libraries
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For plotting and visualizing the data.
- **scikit-learn (sklearn)**: For implementing machine learning models.

### Additional Libraries for Specific Methods
- **tensorflow**: Required for running the Neural Network model.
- **drawdata**: For interactive data point generation used in SVM.

You can install these libraries using the following commands:

\`\`\`bash
pip install pandas numpy matplotlib scikit-learn tensorflow drawdata
\`\`\`

## Data Cleaning and Preprocessing

The data used in this project was first loaded and inspected for any irregularities. The dataset was then cleaned to ensure that all the features were numeric and that there were no missing or invalid entries. The data cleaning process involved:

1. **Conversion of Categorical to Numeric**: All categorical variables were converted into numeric form using label encoding or factorization to ensure they could be used in the machine learning models.
2. **Handling Missing Values**: Any missing values were filled or removed to prevent issues during model training.

These steps ensured that the dataset was ready for training and evaluating the models.

## Classification Methods

### 1. Support Vector Machine (SVM)

**File:** `SVM.py`

The SVM model was implemented to classify data points into different classes. Two variations were used:

- **Hard Margin SVM**: Used for perfectly separable data.
- **Soft Margin SVM**: Used for non-separable data with some misclassifications allowed.

**Results**: The SVM model provided clear decision boundaries, and the classification accuracy was evaluated on the generated datasets.

### 2. Decision Tree

**File:** `Decision_tree.py`

A Decision Tree classifier was implemented to predict the level of lung cancer. The model was trained on the dataset after splitting it into training and testing sets.

**Results**: The Decision Tree model provided a visual representation of the decision process. The accuracy and F1-score of the model were calculated, showing its effectiveness in classification.

### 3. Random Forest and Neural Network

**File:** `Random forest and neural network.ipynb`

This notebook includes the implementation of:

- **Random Forest**: An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
- **Neural Network**: A deep learning model implemented using TensorFlow for more complex pattern recognition.

**Results**: The Random Forest model provided a robust classification with improved accuracy due to the ensemble method. The Neural Network further enhanced the classification performance by capturing more complex relationships in the data.

## Results

- **SVM**: Showed strong performance on linear datasets, with clearly defined decision boundaries.
- **Decision Tree**: Provided good interpretability and accuracy, making it a simple yet effective model.
- **Random Forest**: Enhanced accuracy by reducing overfitting and improving generalization.
- **Neural Network**: Delivered the highest accuracy by leveraging deep learning, though at the cost of interpretability.

## Conclusion

This project demonstrates the effectiveness of different machine learning models in classifying lung cancer levels. Each model has its strengths, from the interpretability of Decision Trees to the high accuracy of Neural Networks. Depending on the specific use case, any of these models can be a valuable tool for predicting lung cancer levels.
