# AI Course Recommendation System

## Project Overview
This project is a simple content-based course recommendation system developed for DecodeLabs Artificial Intelligence Internship Project 3.

The system recommends courses based on the user's interests using TF-IDF vectorization and cosine similarity.

## Features
- Takes user interests as input
- Uses TF-IDF to convert text into numerical vectors
- Uses cosine similarity to compare user interests with course descriptions
- Ranks courses based on similarity
- Displays the top 3 recommendations
- Handles cases where no suitable match is found

## Technologies Used
- Python
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## How It Works
1. Course descriptions are stored in a dataset.
2. TF-IDF converts the course descriptions into vectors.
3. The user's interests are converted into the same vector space.
4. Cosine similarity calculates how closely each course matches the user's interests.
5. Courses are ranked according to their similarity scores.
6. The top 3 recommendations are displayed.

## How to Run

Install the required library:

```bash
pip install scikit-learn

## Author

Khawaja Muhammad Ali Ghauri

Artificial Intelligence Intern — DecodeLabs  
Project 3: AI Recommendation Logic