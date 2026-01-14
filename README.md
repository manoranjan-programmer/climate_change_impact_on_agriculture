# Climate Change Impact on Agriculture

## 🌍 Overview

This project analyzes the impact of climate change on global agriculture through data science and machine learning. It includes a comprehensive dashboard for visualizing trends, forecasting future yields, and simulating climate scenarios to understand how temperature and rainfall changes affect crop productivity.

The project combines exploratory data analysis, predictive modeling, and an interactive web application to provide insights into agricultural sustainability in the face of climate change.

## 📊 Features

### Data Science Analysis (`ds_project.ipynb`)
- **Data Exploration**: Comprehensive analysis of climate and agricultural data
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling
- **Correlation Analysis**: Heatmap visualization of relationships between variables
- **Predictive Modeling**: Linear regression model to predict crop yield based on climate factors
- **Model Evaluation**: Performance metrics including MAE, MSE, RMSE, and R² score
- **Feature Importance**: Analysis of which factors most influence crop yield

### Interactive Dashboard (`app.py`)
- **Global Dashboard**: Summary metrics and trend visualizations
- **Interactive Maps**: Choropleth maps showing yield and temperature by country
- **Future Projections**: Time series forecasting of crop yields
- **Scenario Simulator**: Adjust climate variables to predict yield impacts
- **Filtering Options**: Filter data by country and year

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/climate_change_impact_on_agriculture.git
   cd climate_change_impact_on_agriculture
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present in the project directory:
   - `climate_change_impact_on_agriculture_2024.csv` (dataset)
   - `climate_linear_model.joblib` (trained model)
   - `climate_scaler.joblib` (feature scaler)

## 📈 Usage

### Running the Data Science Notebook
1. Open `ds_project.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells to perform the complete analysis
3. The notebook will generate and save the trained model files

### Running the Dashboard
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`
3. Navigate through the different sections using the sidebar

### Dashboard Sections
- **🏠 Dashboard**: View summary statistics and yield trends
- **🗺️ Maps**: Explore geographical distributions
- **🔮 Projections**: Forecast future crop yields
- **📝 Scenario Simulator**: Simulate different climate scenarios

## 📋 Dataset

The project uses the `climate_change_impact_on_agriculture_2024.csv` dataset containing:
- Country information
- Year of observation
- Average temperature (°C)
- Annual rainfall (mm)
- Crop yield (MT/HA)

## 🤖 Model Details

- **Algorithm**: Linear Regression
- **Target Variable**: Crop Yield (MT/HA)
- **Features**: Temperature, rainfall, year, country (encoded)
- **Preprocessing**: Standard scaling, one-hot encoding for categorical variables

The trained model and scaler are saved as `.joblib` files for use in the dashboard.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Joblib**: Model serialization

## 📁 Project Structure

```
climate_change_impact_on_agriculture/
│
├── app.py                              # Streamlit dashboard application
├── ds_project.ipynb                    # Data science analysis notebook
├── climate_change_impact_on_agriculture_2024.csv  # Dataset
├── climate_linear_model.joblib         # Trained linear regression model
├── climate_scaler.joblib               # Feature scaler
├── requirements.txt                    # Python dependencies
└── README.md                          # Project documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
