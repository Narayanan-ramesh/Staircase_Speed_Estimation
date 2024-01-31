# Staircase Speed Estimation

## Overview

Staircase Speed Estimation is a project designed to estimate the speed of a person ascending or descending stairs from a given input video file using computer vision and machine learning techniques.

## Features

- Speed estimation for both ascending and descending staircases.
- Utilizes computer vision algorithms to track and analyze human movement.
- Machine learning techniques applied for accurate speed predictions.
- Compatible with the HMDB action dataset for robust training and testing.

## Dataset

The project uses the [HMDB action dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) for training and testing purposes.

## Installation

To get started with Staircase Speed Estimation, follow these simple steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/staircase-speed-estimation.git
   ```
   
2. Navigate to the project directory:

   ```bash
   cd staircase-speed-estimation
   ```
   
 3. Ensure you have the HMDB dataset downloaded and copied onto the 'Dataset' folder within the project directory.
 4. Run the following command to install the required packages:
	```bash
	pip install -r requirements.txt
	```

## Usage
After completing the installation, you can run the main script to start estimating staircase speed. Execute the following command:

```bash
python main.py
```