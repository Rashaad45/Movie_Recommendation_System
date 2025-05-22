from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load the dataset (Ensure the path is correct)
movies = pd.read_csv('Copy of Full_Movie_Dataset(1).csv')
movies.isnull().sum()
# Remove duplicates
movies = movies.drop_duplicates()

# Feature Extraction
# Create TFIDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# Transform the descriptions into TFIDF vectors
tfidf_matrix = tfidf.fit_transform(movies['Description'])
description_movies = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Genre encoding
genre_encoded = movies['Genre'].str.get_dummies(sep=',')

# Label encoding for ACTORS
actor_encoder = LabelEncoder()
actor_encoded = actor_encoder.fit_transform(movies['Actors']).reshape(-1, 1)

# Scaling RATINGS
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(movies[['Rating']])

# Final feature matrix formation
# Ensure all arrays are 2D
genre_encoded = genre_encoded.reshape(-1, 1) if len(genre_encoded.shape) == 1 else genre_encoded
scaled_features = scaled_features.reshape(-1, 1) if len(scaled_features.shape) == 1 else scaled_features
feature_matrix = np.hstack((description_movies, genre_encoded, actor_encoded, scaled_features))
feature_movies = pd.DataFrame(feature_matrix, index=movies['Title'])

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(feature_movies)

# KMeans Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(reduced_vectors)
movies["Clusters"] = kmeans.labels_

def recommend_movies(search_title):
    search_movie = movies[movies["Title"].str.lower() == search_title.lower()]
    if search_movie.empty:
        return {"error": "Movie not found in dataset"}, None
    
    # Transform the search data
    search_index = search_movie.index[0]
    search_feature_pca = reduced_vectors[search_index].reshape(1, -1)
    
    # Cosine similarity
    similarities = cosine_similarity(search_feature_pca, reduced_vectors).flatten()
    
    # KMeans prediction of cluster
    predicted_clusters = kmeans.predict(search_feature_pca)
    
    # Add similarities to movies dataframe
    movies["similarity"] = similarities
    
    # Get recommended movies
    recommended_movies = movies[movies["Clusters"] == predicted_clusters[0]].copy()
    recommended_movies = recommended_movies.sort_values(by="similarity", ascending=False)
    
    # Return the searched movie and recommendations
    searched_movie_info = search_movie[["Title", "Rating"]].to_dict(orient='records')[0]
    recommendations_list = recommended_movies[["Title", "similarity", "Rating"]].iloc[1:11].to_dict(orient='records')
    
    return searched_movie_info, recommendations_list

# Django view
def genai(request):
    if request.method == 'POST':
        search_query = request.POST.get("num1", "")
        if not search_query:
            return render(request, 'genai.html', {"error": "No movie title provided"})
        
        searched_movie, recommendations = recommend_movies(search_query)
        
        if "error" in searched_movie:
            return render(request, 'genai.html', {"error": searched_movie["error"]})
        
        context = {
            "searched_movie": searched_movie,
            "recommendations": recommendations
        }
        
        return render(request, 'genai.html', context)
    else:
        # Handle GET request
        return render(request, 'genai.html')
    

def form(request):
    return render(request,'form.html')

def AI(request):
    search_results = []
    
    if request.method == "POST":
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key="AIzaSyA96OScR17VpKckBCr-NHDm7J8-v0Ed6Uc")
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Get the query from the form
        query = request.POST.get('query', '')
        
        if query:
            # Generate response from Gemini
            response = model.generate_content(query)
            
            # Process the response - split by lines or other delimiter as needed
            search_results = response.text.split('\n')
            # Filter out empty lines
            search_results = [line for line in search_results if line.strip()]
    
    # Pass the results to the template
    return render(request, 'AI.html', {"search": search_results})