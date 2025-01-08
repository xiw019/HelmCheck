# HelmCheck — A Computer Vision–Based Global Helmet Wearing Rate Sampling Method

This project integrates the Google Street View API and a custom-trained YOLOv5 model to sample, detect, classify, and analyze helmet-wearing rates of motorcyclists (both passengers and drivers) in different cities. The workflow includes:

1. Generating GPS coordinates along specified routes.
2. Downloading Google Street View images for those coordinates.
3. Using a pretrained YOLOv5 model to detect and classify drivers, passengers, and helmets.
4. Calculating helmet-wearing metrics and summarizing findings.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Running Environment](#running-environment)
  - [Software Requirements](#software-requirements)
  - [Python Packages](#python-packages)
  - [Additional Tools](#additional-tools)
  - [PyTorch Environment](#pytorch-environment)
- [Notebook Details](#notebook-details)
  - [1. `sample_images.ipynb`](#1-sample_imagesipynb)
  - [2. `get_metric.ipynb`](#2-get_metricipynb)
  - [3. `metric_summary.ipynb`](#3-metric_summaryipynb)
- [Usage Notes](#usage-notes)
- [Contact](#contact)

---

## Project Overview
The goal of this project is to automate the process of assessing helmet usage across multiple cities. It helps collect and label data about motorcyclists on the road, enabling large-scale studies of compliance rates regarding helmet use. By leveraging computer vision and the Google Street View API, the process becomes reproducible and scalable.

---

## Project Structure
This repository contains:

- **Notebooks**:
  - `sample_images.ipynb` — for routing and downloading Street View images.  
  - `get_metric.ipynb` — for YOLO-based object detection and helmet-wearing analysis.  
  - `metric_summary.ipynb` — for consolidating city-level data into a single summary.
- **Model File**:  
  - `wear_metric.pt` (a pretrained YOLOv5 model) — for detecting helmets, drivers, and passengers.
- **Metadata & Image Files**:  
  - CSV files for storing metadata (coordinates, timestamps, detection results, etc.).
  - Directories containing downloaded or cropped images (used temporarily during processing).

---

## Running Environment

### Software Requirements
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Python Packages
- `pandas`
- `numpy`
- `torch`
- `opencv-python`
- `Pillow`
- `requests`
- `folium`
- `re`
- `polyline`
- `csv`

### Additional Tools
- **Pretrained YOLO Model**: Place `wear_metric.pt` in the designated directory (used in `get_metric.ipynb`).
- **Google Street View API Key**: Required in `sample_images.ipynb` for fetching Street View images.

### PyTorch Environment
- Ensure that PyTorch is installed based on your system’s hardware configuration. For systems with GPU support, install the CUDA-compatible version of PyTorch to leverage faster inference. Refer to the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for detailed instructions.
- Suggested Environment Setup:
  - Install `torch` and `torchvision` using:
    ```
    pip install torch torchvision
    ```
  - Verify installation by running:
    ```
    python -c "import torch; print(torch.cuda.is_available())"
    ```

---

## Notebook Details

### 1. `sample_images.ipynb`
**Description**  
This notebook handles the downloading of images and associated metadata using route coordinates generated via the OSRM API.

**Key Features**
- **Route Processing**:
  - Generates GPS coordinates for routes using the OSRM API.
- **Image Download**:
  - Fetches Street View images based on the generated GPS coordinates.
  - Saves images locally and records metadata (latitude, longitude, timestamp) in CSV.
- **Batch Processing**:
  - Automates route processing to achieve a target number of images per route or per city.

**Outputs**
- Downloaded Street View images organized by city.
- Metadata CSV files for each city (with image names, timestamps, and GPS coordinates).

### 2. `get_metric.ipynb`
**Description**  
This notebook includes scripts for object detection and metadata updates using a YOLO model. It processes cropped images to extract relevant information (drivers, passengers, helmets) and stores the detections in a structured format.

**Key Features**
- **Object Detection**:
  - Identifies objects (e.g., helmets, drivers, passengers) in images using a pretrained YOLOv5 model (`wear_metric.pt`).
  - Converts bounding boxes to YOLO format if needed.
- **Metadata Management**:
  - Updates metadata CSV files for individual cities.
  - Iteratively processes multiple cities, appending or updating the relevant columns (e.g., helmet usage).
- **Helmet Usage Analysis**:
  - Analyzes helmet usage rates for drivers and passengers.
  - Outputs metrics, including percentages of compliance.

**Outputs**
- Updated metadata CSV files (with detection results and helmet usage flags).
- Summarized metrics for helmet-wearing compliance.

### 3. `metric_summary.ipynb`
**Description**  
This notebook consolidates and summarizes metadata across multiple files and cities, creating a single dataset suitable for broader statistical analysis or visualization.

**Key Features**
- **Aggregation**:
  - Aggregates individual city-level CSV files into a single consolidated DataFrame.
- **Metadata Enhancement**:
  - Adds contextual columns (e.g., city names).
- **Export**:
  - Exports the combined dataset to a single CSV file for further analysis (`combined_metadata.csv`).

**Outputs**
- A single `combined_metadata.csv` file containing consolidated metadata and helmet usage summaries from multiple cities.

---

## Usage Notes

- **Performance**:
  - GPU-based inference in PyTorch will significantly speed up YOLO detection if a GPU is available.
- **Customization**:
  - You can modify route generation, detection thresholds, or city selection as needed.
- **Data Management**:
  - Keep track of storage space for large image downloads and model weights.
  - Temporary folders used for downloads can be cleaned up after processing to save storage space.

---

## Contact
For any questions or feedback, please contact qli28@jh.edu or xwang344@jh.edu