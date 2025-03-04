# Student Performance Prediction

## Overview
This Streamlit application predicts student performance based on various factors such as study habits, learning preferences, and lifestyle choices. The model uses a Random Forest Classifier trained on synthetic student data.

## Features
- **Interactive UI:** Built with Streamlit for ease of use.
- **Data Generation:** Generates synthetic data for demonstration purposes.
- **Data Preprocessing:** Handles categorical encoding and standardization.
- **Machine Learning Model:** Implements a Random Forest Classifier for prediction.
- **Feature Importance Visualization:** Displays feature importance using a bar chart.
- **Model Performance Metrics:** Shows accuracy and classification report.
- **Real-time Predictions:** Accepts user input and predicts final grade.
- **Dynamic Background:** Uses custom CSS and JavaScript for animated background effects.

## Installation
To run this application locally, follow these steps:

### Prerequisites
Ensure you have Python installed. You also need to install the required libraries.

```sh
pip install streamlit pandas numpy seaborn scikit-learn matplotlib
```

### Creating a Virtual Environment
It is recommended to create a virtual environment before installing dependencies.

```sh
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### Running the Application
Clone this repository and navigate to the project folder:

```sh
git clone <repository-url>
cd <project-folder>
```

Run the Streamlit app:

```sh
streamlit run app.py
```

## Usage
1. Launch the app using the command above.
2. Explore the dataset by selecting "Show Dataset."
3. View feature importance and model performance metrics.
4. Enter custom student data and get a predicted final grade.

## File Structure
- `app.py` - Main Streamlit application script.
- `requirements.txt` - List of dependencies.
- `README.md` - Documentation for the project.

## Technologies Used
- **Python** for scripting.
- **Streamlit** for UI development.
- **Scikit-learn** for machine learning.
- **Pandas & NumPy** for data handling.
- **Seaborn & Matplotlib** for data visualization.
- **HTML, CSS, JavaScript** for UI enhancements.

## License
This project is open-source and available under the MIT License.

## Live Application
Link-https://student-performance-prediction-saaho-shanjai.streamlit.app

