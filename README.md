# 🔵 Day 36 – DBSCAN Clustering on Starbucks Geolocation Data
### 100 Days AI/ML Engineer Challenge
 
> *"If you use K-Means for geolocation clustering without thinking, you're forcing structure that doesn't exist."*
 
---
 
## 📌 Overview
 
This project implements **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on the **Starbucks Store Locations Dataset** to identify regional clusters of stores worldwide based on latitude and longitude.
 
Unlike K-Means, DBSCAN doesn't require you to specify the number of clusters — it discovers them purely from data density.
 
---
 
## 🧠 What is DBSCAN?
 
DBSCAN groups data points based on how densely packed they are in a region. It has two key parameters:
 
| Parameter | Description |
|---|---|
| `ε (epsilon)` | Radius of the neighborhood around a point |
| `min_samples` | Minimum points needed to form a dense region |
 
### Point Types
 
| Type | Description |
|---|---|
| **Core Point** | Has at least `min_samples` neighbors within `ε` |
| **Border Point** | Within `ε` of a core point, but not dense enough itself |
| **Noise Point** | Not reachable from any core point — treated as outlier |
 
---
 
## 📊 Dataset
 
**Starbucks Store Locations Dataset**
- Features used: `Latitude`, `Longitude`
- Global store location data
- Source: [Kaggle – Starbucks Locations Worldwide](https://www.kaggle.com/datasets/starbucks/store-locations)
 
---
 
## 🔧 Implementation Steps
 
### 1. Data Preprocessing
- Extracted `latitude` and `longitude` columns
- Dropped rows with missing or invalid coordinate values
- Converted coordinates to radians for Haversine distance metric
 
### 2. DBSCAN Application
```python
from sklearn.cluster import DBSCAN
import numpy as np
 
coords = df[['Latitude', 'Longitude']].values
coords_rad = np.radians(coords)
 
db = DBSCAN(eps=0.5, min_samples=5, algorithm='ball_tree', metric='haversine')
df['Cluster'] = db.fit_predict(coords_rad)
```
 
### 3. Cluster Analysis
- Identified dense regional clusters (high store concentration)
- Detected sparse regions and isolated store locations
- Labeled noise points as `-1`
 
### 4. Visualization
```python
import matplotlib.pyplot as plt
 
plt.figure(figsize=(14, 7))
scatter = plt.scatter(df['Longitude'], df['Latitude'],
                      c=df['Cluster'], cmap='tab20', s=1, alpha=0.6)
plt.title('DBSCAN Clustering – Starbucks Store Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.show()
```
 
---
 
## 📈 Key Insights
 
- DBSCAN naturally discovered clusters aligned with real-world geography (US East Coast, Western Europe, East Asia)
- No need to pre-define `K` — clusters emerged from density alone
- Isolated stores in low-density regions were correctly flagged as **noise**
- Cluster shapes are **irregular**, unlike K-Means' circular boundaries
 
---
 
## ⚠️ Limitations
 
- Sensitive to `eps` and `min_samples` — requires tuning
- Struggles with datasets of **varying density**
- Latitude/longitude need proper distance scaling (Haversine recommended over Euclidean)
 
---
 
## 📦 Requirements
 
```
pandas
numpy
scikit-learn
matplotlib
```
 
Install with:
```bash
pip install pandas numpy scikit-learn matplotlib
```
 
---
 
## 🆚 DBSCAN vs K-Means — Quick Comparison
 
| Feature | DBSCAN | K-Means |
|---|---|---|
| Number of clusters | Auto-detected | Must specify K |
| Handles noise | ✅ Yes | ❌ No |
| Cluster shape | Any shape | Spherical only |
| Geospatial data | ✅ Great fit | ⚠️ Poor fit |
| Varying density | ⚠️ Struggles | ❌ Struggles |
 
---
 
## 📁 Project Structure
 
```
day36-dbscan-clustering/
├── dbscan_starbucks.ipynb   # Main notebook
├── starbucks_locations.csv  # Dataset
├── cluster_map.png          # Visualization output
└── README.md
```
 
---
 
## 🚀 Part of the 100 Days AI/ML Engineer Challenge
 
| Day | Topic |
|---|---|
| Day 34 | K-Means Clustering |
| Day 35 | Hierarchical Clustering |
| **Day 36** | **DBSCAN – Density-Based Clustering** |
| Day 37 | Coming up... |
 
---
 
*Follow the journey on [LinkedIn](https://www.linkedin.com/) · #100DaysAIML #DBSCAN #UnsupervisedLearning*
