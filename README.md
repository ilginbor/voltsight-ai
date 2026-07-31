# VoltSight

**Explainable AI for intelligent EV charging station site selection and urban infrastructure planning.**

VoltSight is an explainable artificial intelligence project that identifies suitable areas for new electric vehicle charging stations using geospatial data and machine learning.

## Initial Study Area

The first version of VoltSight focuses on Ã‡ankaya, Ankara.

The study area will be divided into 250 Ã— 250 meter grid cells. Each grid cell will be evaluated using spatial and urban features.

## Project Goals

- Collect and process open geospatial datasets
- Generate 250 Ã— 250 meter analysis grids
- Create spatial machine learning features
- Train and compare baseline and advanced models
- Generate charging station suitability scores
- Explain model predictions using SHAP
- Estimate prediction uncertainty
- Serve predictions through a FastAPI backend
- Visualize results using React and OpenLayers

## Planned Data Features

- Distance to main roads
- Road density
- Nearby parking areas
- Existing charging station density
- Distance to existing charging stations
- Commercial and residential density
- Points of interest
- Population and urban density indicators
- Distance to electrical infrastructure

## Planned Technologies

### Data Science and Machine Learning

- Python
- Pandas
- NumPy
- GeoPandas
- Scikit-learn
- XGBoost
- SHAP

### Backend and Data Storage

- FastAPI
- PostgreSQL
- PostGIS

### Frontend and Visualization

- React
- OpenLayers

### Engineering and MLOps

- Git and GitHub
- Docker
- MLflow
- Pytest

## Project Structure

```text
voltsight-ai/
|-- backend/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- docs/
|-- frontend/
|-- notebooks/
|-- src/
|   `-- voltsight/
|       |-- data/
|       |-- evaluation/
|       |-- features/
|       `-- models/
|-- tests/
|-- .gitignore
|-- README.md
`-- requirements.txt
```