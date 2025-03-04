import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# css bg
st.markdown("""
<style>
.stApp {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    background-size: 400% 400%;
    animation: gradient 6s ease infinite;
    transition: background 3s ease;
}

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
</style>
<script>
const gradients = [
    'linear-gradient(45deg, #ff6b6b, #4ecdc4)',
    'linear-gradient(45deg, #a8e6cf, #dcedc1)',
    'linear-gradient(45deg, #ffd3b6, #ffaaa5)',
    'linear-gradient(45deg, #b8c1ec, #e2d1f9)',
];

let currentIndex = 0;

function changeGradient() {
    const app = document.querySelector('.stApp');
    currentIndex = (currentIndex + 1) % gradients.length;
    app.style.background = gradients[currentIndex];
}

setInterval(changeGradient, 3000);
</script>
""", unsafe_allow_html=True)



# Initialize session
if 'model' not in st.session_state:
    st.session_state.model = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None

    
# Load dataset 
@st.cache_data
def generate_data(n=1000):  
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 31, n),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n),
        'Study_Hours_per_Week': np.random.randint(5, 51, n),
        'Preferred_Learning_Style': np.random.choice(['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'], n),
        'Online_Courses_Completed': np.random.randint(0, 21, n),
        'Participation_in_Discussions': np.random.choice(['Yes', 'No'], n),
        'Assignment_Completion_Rate': np.random.randint(50, 101, n),
        'Exam_Score': np.random.randint(40, 101, n),
        'Attendance_Rate': np.random.randint(50, 101, n),
        'Use_of_Educational_Tech': np.random.choice(['Yes', 'No'], n),
        'Self_Reported_Stress_Level': np.random.choice(['Low', 'Medium', 'High'], n),
        'Time_Spent_on_Social_Media': np.random.randint(0, 31, n),
        'Sleep_Hours_per_Night': np.random.randint(4, 11, n)
    }
    df = pd.DataFrame(data)
    df['Final_Grade'] = pd.cut(df['Exam_Score'], bins=[40, 50, 60, 70, 80, 100], labels=['F', 'D', 'C', 'B', 'A'])
    return df

# Main application
st.title("Student Performance Prediction")
st.write("This app predicts student performance based on various factors.")

# Add sidebar for better organization
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Performance", "Make Prediction"])

if page == "Data Overview":
    st.header("Dataset Information")
    if st.session_state.df is None:
        st.write("Generating and preprocessing data...")
        df = generate_data()
        
        # Preprocessing
        label_encoders = {}
        categorical_cols = ['Gender', 'Preferred_Learning_Style', 'Participation_in_Discussions', 
                           'Use_of_Educational_Tech', 'Self_Reported_Stress_Level']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        df['Final_Grade'] = df['Final_Grade'].astype(str)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.label_encoders = label_encoders

        # Train model
        X = df.drop(columns=['Final_Grade'])
        y = df['Final_Grade']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.session_state.model = model
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        st.session_state.accuracy = accuracy_score(y_test, y_pred)
        st.session_state.classification_report = classification_report(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", st.session_state.df.shape)
    with col2:
        st.write("Number of Classes:", len(st.session_state.df['Final_Grade'].unique()))
    
    if st.checkbox("Show Dataset"):
        st.dataframe(st.session_state.df)
    
    # Add data visualization
    st.subheader("Data Distribution")
    selected_column = st.selectbox("Select column to visualize", st.session_state.df.columns)
    fig, ax = plt.subplots()
    if st.session_state.df[selected_column].dtype in ['int64', 'float64']:
        sns.histplot(data=st.session_state.df, x=selected_column, hue='Final_Grade')
    else:
        sns.countplot(data=st.session_state.df, x=selected_column, hue='Final_Grade')
    st.pyplot(fig)

elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    # Feature Importance
    st.subheader("Feature Importance")
    if st.session_state.model is not None:
        feature_importance = pd.Series(
            st.session_state.model.feature_importances_,
            index=st.session_state.df.drop(columns=['Final_Grade']).columns
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Model metrics
    if hasattr(st.session_state, 'accuracy'):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
        with col2:
            st.text("Classification Report:")
            st.text(st.session_state.classification_report)

else:  # Make Prediction
    st.header("Make a Prediction")
    
    # Create columns for better input organization
    col1, col2 = st.columns(2)
    user_input = {}
    X_cols = st.session_state.df.drop(columns=['Final_Grade']).columns
    
    for i, col in enumerate(X_cols):
        with col1 if i % 2 == 0 else col2:
            if col in st.session_state.label_encoders:
                user_input[col] = st.selectbox(
                    col, 
                    st.session_state.label_encoders[col].classes_
                )
            else:
                user_input[col] = st.number_input(
                    col, 
                    float(st.session_state.df[col].min()), 
                    float(st.session_state.df[col].max()), 
                    float(st.session_state.df[col].mean()),
                    help=f"Range: {st.session_state.df[col].min():.0f} - {st.session_state.df[col].max():.0f}"
                )

    if st.button("Predict Final Grade", type="primary"):
        input_df = pd.DataFrame([user_input])
        
        for col in st.session_state.label_encoders:
            input_df[col] = st.session_state.label_encoders[col].transform(input_df[col])
        
        prediction = st.session_state.model.predict(input_df)[0]
        prediction_proba = st.session_state.model.predict_proba(input_df)[0]
        max_proba = max(prediction_proba)
        
        st.success(f"Predicted Final Grade: {prediction} (Confidence: {max_proba:.2%})")

st.write("Debug: Application loaded successfully")