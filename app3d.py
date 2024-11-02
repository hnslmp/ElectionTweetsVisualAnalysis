# Install necessary packages (if not already installed)
# Uncomment the following lines to install required packages
# !pip install -U sentence-transformers
# !pip install umap-learn
# !pip install plotly
# !pip install nltk
# !pip install scikit-learn
# !pip install minisom
# !pip install dash
# !pip install dash-bootstrap-components
# !pip install vaderSentiment
# !pip install statsmodels

import os
import pandas as pd
import numpy as np
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter  # For word frequency
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import umap
from minisom import MiniSom
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.exceptions import PreventUpdate
import plotly.io as pio  # For figure serialization/deserialization
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.nonparametric.smoothers_lowess import lowess 
from sklearn.preprocessing import MinMaxScaler 
import random 

# ------------------- Caching Functions -------------------

def save_embeddings(embeddings, filename='embeddings.npy'):
    """
    Saves embeddings to a .npy file.
    """
    np.save(filename, embeddings)
    print(f"Embeddings saved to {filename}.")

def load_embeddings(filename='embeddings.npy'):
    """
    Loads embeddings from a .npy file.
    """
    if os.path.exists(filename):
        embeddings = np.load(filename)
        print(f"Embeddings loaded from {filename}.")
        return embeddings
    else:
        print(f"No cached embeddings found at {filename}.")
        return None

# ------------------- Data Loading and Preprocessing -------------------

# Download NLTK data (ensure this is done before running the code)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

# Load your data
df = pd.read_csv('tweets.csv')
tweets = df['text'].astype(str)  # Ensure all tweets are strings

# Ensure 'created_at' or 'time' is datetime
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    # Create a 'date' column for grouping
    df['date'] = df['created_at'].dt.date
elif 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # Create a 'date' column for grouping
    df['date'] = df['time'].dt.date
else:
    # Handle the case where neither 'created_at' nor 'time' exist
    print("No timestamp column ('created_at' or 'time') found in the dataset.")
    df['date'] = pd.to_datetime('today').date()  # Assign current date to all

# Ensure engagement columns are numeric and fill NaN values with zero
df['retweet_count'] = pd.to_numeric(df.get('retweet_count', 0), errors='coerce').fillna(0)
df['favorite_count'] = pd.to_numeric(df.get('favorite_count', 0), errors='coerce').fillna(0)

# Calculate engagement
df['engagement'] = df['retweet_count'] + df['favorite_count']

# Apply logarithmic transformation to engagement
df['engagement_log'] = np.log1p(df['engagement'])

# Ensure 'handle' column exists and is of type string
if 'handle' not in df.columns:
    df['handle'] = 'Unknown'  # Replace with appropriate default or data source
df['handle'] = df['handle'].astype(str)

# Preprocess the tweets with lemmatization
def preprocess_tweet(text):
    """
    Preprocesses a tweet by removing URLs, mentions, hashtags, special characters,
    converting to lowercase, removing stopwords, and lemmatizing words.
    """
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|@\S+|#\S+|[^A-Za-z0-9\s]+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Rejoin words
    return ' '.join(words)

tweets_cleaned = tweets.apply(preprocess_tweet)

# Remove tweets that are too short (less than 3 words)
df['tweet_length'] = tweets_cleaned.apply(lambda x: len(x.split()))
df = df[df['tweet_length'] >= 3]
tweets_cleaned = df['text'].apply(preprocess_tweet)
df = df.reset_index(drop=True)
tweets_cleaned = tweets_cleaned.reset_index(drop=True)

# ------------------- Embedding with Caching -------------------

# Define the embeddings file path
embeddings_file = 'embeddings.npy'

# Attempt to load cached embeddings
embeddings = load_embeddings(embeddings_file)

if embeddings is None:
    # If not cached, compute embeddings
    print("Computing embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(tweets_cleaned.to_list(), show_progress_bar=True)
    # Save embeddings for future use
    save_embeddings(embeddings, embeddings_file)
else:
    # If loaded from cache, skip embedding computation
    print("Using cached embeddings.")

# ------------------- Sentiment Analysis -------------------

# Use VADER to calculate sentiment scores
analyzer = SentimentIntensityAnalyzer()
sentiment_scores_vader = []
for tweet in tweets_cleaned:
    sentiment = analyzer.polarity_scores(tweet)
    sentiment_scores_vader.append(sentiment['compound'])

# Create sentiment DataFrame including Engagement, Handle, and Date
sentiment_df = pd.DataFrame({
    'Index': df.index,  # Ensure alignment with df
    'Sentiment Score': sentiment_scores_vader,
    'Engagement': df['engagement'],
    'Handle': df['handle'],
    'Date': df['date']  # Use the correct 'date' column
})
sentiment_df.index = df.index  # Ensure indices match

# ------------------- Dimensionality Reduction -------------------

# Optionally reduce dimensionality with PCA before t-SNE/UMAP
pca_file = 'embeddings_pca.npy'
embeddings_pca = None

if os.path.exists(pca_file):
    embeddings_pca = np.load(pca_file)
    print(f"PCA-transformed embeddings loaded from {pca_file}.")
else:
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    np.save(pca_file, embeddings_pca)
    print(f"PCA-transformed embeddings saved to {pca_file}.")

# Standardize the embeddings
scaler_file = 'embeddings_scaled.npy'
embeddings_scaled = None

if os.path.exists(scaler_file):
    embeddings_scaled = np.load(scaler_file)
    print(f"Standardized embeddings loaded from {scaler_file}.")
else:
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_pca)
    np.save(scaler_file, embeddings_scaled)
    print(f"Standardized embeddings saved to {scaler_file}.")

# Perform t-SNE
tsne_file = 'embeddings_tsne.npy'
embeddings_2d_tsne = None

if os.path.exists(tsne_file):
    embeddings_2d_tsne = np.load(tsne_file)
    print(f"t-SNE embeddings loaded from {tsne_file}.")
else:
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42
    )
    embeddings_2d_tsne = tsne.fit_transform(embeddings_scaled)
    np.save(tsne_file, embeddings_2d_tsne)
    print(f"t-SNE embeddings saved to {tsne_file}.")

# Perform UMAP
umap_file = 'embeddings_umap.npy'
embeddings_2d_umap = None

if os.path.exists(umap_file):
    embeddings_2d_umap = np.load(umap_file)
    print(f"UMAP embeddings loaded from {umap_file}.")
else:
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    embeddings_2d_umap = reducer.fit_transform(embeddings_scaled)
    np.save(umap_file, embeddings_2d_umap)
    print(f"UMAP embeddings saved to {umap_file}.")

# Create DataFrames for visualization
vis_df_tsne = pd.DataFrame({
    'x': embeddings_2d_tsne[:, 0],
    'y': embeddings_2d_tsne[:, 1],
    'tweet': tweets_cleaned,
    'engagement': df['engagement'],
    'engagement_log': df['engagement_log'],
    'date': df['date']
})
vis_df_tsne.index = df.index  # Ensure indices match

vis_df_umap = pd.DataFrame({
    'x': embeddings_2d_umap[:, 0],
    'y': embeddings_2d_umap[:, 1],
    'tweet': tweets_cleaned,
    'engagement': df['engagement'],
    'engagement_log': df['engagement_log'],
    'date': df['date']
})
vis_df_umap.index = df.index  # Ensure indices match

# ------------------- SOM Training -------------------

# Define SOM parameters and files
som_grid_size = 40  # Grid size (40x40)
som_file = 'som.npy'
distance_map_file = 'distance_map.npy'

# Initialize SOM
if os.path.exists(som_file) and os.path.exists(distance_map_file):
    # Load existing SOM weights and distance map
    print(f"Loading SOM from {som_file} and distance map from {distance_map_file}.")
    som = MiniSom(som_grid_size, som_grid_size, embeddings_scaled.shape[1],
                  sigma=1.0, learning_rate=0.5, neighborhood_function='gaussian', random_seed=42)
    som._weights = np.load(som_file)
    distance_map = np.load(distance_map_file)
else:
    # Train SOM
    print("Training SOM...")
    som = MiniSom(
        x=som_grid_size,
        y=som_grid_size,
        input_len=embeddings_scaled.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )
    som.random_weights_init(embeddings_scaled)
    som.train_random(embeddings_scaled, num_iteration=1000)
    print("SOM training completed.")
    
    # Save SOM weights
    np.save(som_file, som._weights)
    print(f"SOM weights saved to {som_file}.")
    
    # Compute distance map (U-Matrix)
    distance_map = som.distance_map()
    np.save(distance_map_file, distance_map)
    print(f"Distance map saved to {distance_map_file}.")

distance_map_df = pd.DataFrame(distance_map).iloc[::-1]  # Invert y-axis for visualization

# ------------------- Mapping Tweets to SOM Grid Cells -------------------

# Get the coordinates for each data point
coords = np.array([som.winner(x) for x in embeddings_scaled])

# Normalize the coordinates
coords_normalized = coords.astype(float)
coords_normalized[:, 0] /= som_grid_size
coords_normalized[:, 1] /= som_grid_size

# Create DataFrame for SOM visualization
vis_df_som = pd.DataFrame({
    'x': coords_normalized[:, 0],
    'y': coords_normalized[:, 1],
    'som_x': coords[:, 0],
    'som_y': coords[:, 1],
    'tweet': tweets_cleaned,
    'engagement': df['engagement'],
    'engagement_log': df['engagement_log'],
    'date': df['date']
})
vis_df_som.index = df.index  # Ensure indices match

# Map tweets to SOM grid cells with tweet indices
som_grid_df = pd.DataFrame({
    'x': coords[:, 0],
    'y': coords[:, 1],
    'tweet': tweets_cleaned,
    'index': df.index,  # Keep track of tweet indices
    'date': df['date']
})
som_grid_df.index = df.index  # Ensure indices match

# Group tweets by their grid cell coordinates
grouped = som_grid_df.groupby(['x', 'y'])['index'].apply(list).reset_index()

# Create a mapping from grid cell to tweet indices
grid_to_tweets = {}
for _, row in grouped.iterrows():
    x = row['x']
    y = row['y']
    indices = row['index']
    grid_to_tweets[(x, y)] = indices

# ------------------- Create 3D Engagement Map -------------------

# Initialize total_engagement_map with zeros, same shape as distance_map
total_engagement_map = np.zeros_like(distance_map)

# Iterate through each grid cell and sum engagements
for (x, y), indices in grid_to_tweets.items():
    # Sum engagement for all tweets in the current cell
    total_engagement = df.loc[indices, 'engagement'].sum()
    total_engagement_map[y, x] = total_engagement  # Note: y corresponds to rows, x to columns

# Use the original engagement values for z_data
z_data = total_engagement_map

# Convert distance_map_df to NumPy array for surfacecolor
z_data_original = distance_map_df.values

# Create the 3D surface plot
surface = go.Surface(
    z=z_data,
    x=list(range(z_data.shape[1])),  # SOM X-axis
    y=list(range(z_data.shape[0])),  # SOM Y-axis
    surfacecolor=z_data_original,
    colorscale='YlOrRd',  # Choose a perceptually uniform colorscale
    colorbar=dict(title='Distance Map Value'),
    hovertemplate='SOM X: %{x}<br>SOM Y: %{y}<br>Engagement: %{z:.2f}<br>Distance: %{surfacecolor:.4f}<extra></extra>'
)

# Define the layout for the 3D plot
layout_3d = go.Layout(
    title='SOM Engagement Map - 3D Visualization',
    scene=dict(
        xaxis_title='SOM X',
        yaxis_title='SOM Y',
        zaxis_title='Engagement',
        xaxis=dict(nticks=10),
        yaxis=dict(nticks=10),
        zaxis=dict(nticks=5),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera angle as needed
        )
    ),
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=50)
)

# Create the figure
fig_3d = go.Figure(data=[surface], layout=layout_3d)

# ------------------- Create 2D U-Matrix Heatmap -------------------

# Define the heatmap with an enhanced colorscale
heatmap = go.Heatmap(
    z=distance_map_df.values,
    x=list(range(distance_map_df.shape[1])),
    y=list(range(distance_map_df.shape[0])),
    colorscale='YlOrRd',  # Choose a clear colorscale
    colorbar=dict(title='Distance'),
    hoverinfo='text',
    hovertemplate='<b>SOM X: %{x}</b><br><b>SOM Y: %{y}</b><br>Distance: %{z}<extra></extra>'
)

# Create the distance map figure
fig_distance_map = go.Figure(data=[heatmap])

# Add grid lines to the heatmap
fig_distance_map.update_layout(
    title='SOM Distance Map (U-Matrix) - 2D Visualization',
    xaxis_title='SOM X',
    yaxis_title='SOM Y',
    yaxis=dict(autorange='reversed'),  # To match the DataFrame inversion
    shapes=[
        dict(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=som_grid_size,
            line=dict(color="black", width=1)
        ) for x in range(som_grid_size + 1)
    ] + [
        dict(
            type="line",
            x0=0,
            y0=y,
            x1=som_grid_size,
            y1=y,
            line=dict(color="black", width=1)
        ) for y in range(som_grid_size + 1)
    ],
    hovermode='closest'
)


# ------------------- CRITICAL MOVEMENT PREPROCESSING (START) -------------------

# ------------------- Sentiment Over Date -------------------

# Minimum and maximum sentiment scores for both candidates
min_score = min(sentiment_df['Sentiment Score'])
max_score = max(sentiment_df['Sentiment Score'])

def normalize_sentiment(score, min_score, max_score):
    """Normalize the sentiment score to the range [-1, 1]."""
    if max_score == min_score:  # Prevent division by zero
        return 0  # or some other appropriate handling
    normalized_score = 2 * (score - min_score) / (max_score - min_score) - 1
    return normalized_score

# Ensure the 'Date' column is in datetime format
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# Normalize the sentiment scores for Clinton
sentiment_df['Hillary_Net_Sentiment'] = sentiment_df[sentiment_df['Handle'] == 'HillaryClinton']['Sentiment Score'].apply(
    lambda x: normalize_sentiment(x, min_score, max_score)
)

# Normalize the sentiment scores for Trump
sentiment_df['Trump_Net_Sentiment'] = sentiment_df[sentiment_df['Handle'] == 'realDonaldTrump']['Sentiment Score'].apply(
    lambda x: normalize_sentiment(x, min_score, max_score)
)

# Filter for tweets related to Clinton and Trump
clinton_tweets = sentiment_df[sentiment_df['Handle'] == 'HillaryClinton']
trump_tweets = sentiment_df[sentiment_df['Handle'] == 'realDonaldTrump']

# Set the date column as the index for resampling
clinton_tweets.set_index('Date', inplace=True)
trump_tweets.set_index('Date', inplace=True)

# Resample to daily normalized sentiment scores
hillary_daily_sentiment = clinton_tweets['Hillary_Net_Sentiment'].resample('D').mean().reset_index()
trump_daily_sentiment = trump_tweets['Trump_Net_Sentiment'].resample('D').mean().reset_index()

# Rename the columns for clarity
hillary_daily_sentiment.rename(columns={'Hillary_Net_Sentiment': 'Hillary_Net_Sentiment'}, inplace=True)
trump_daily_sentiment.rename(columns={'Trump_Net_Sentiment': 'Trump_Net_Sentiment'}, inplace=True)

# Merge the two DataFrames on the date column
daily_net_sentiment = pd.merge(hillary_daily_sentiment, trump_daily_sentiment, on='Date', how='outer')

# ------------------- Linear Sentiment Over Date -------------------

# Smooth Hillary Clinton's sentiment scores using the original dates
hillary_smoothed = lowess(daily_net_sentiment['Hillary_Net_Sentiment'], 
                           daily_net_sentiment['Date'], 
                           frac=0.1)

# Smooth Donald Trump's sentiment scores using the original dates
trump_smoothed = lowess(daily_net_sentiment['Trump_Net_Sentiment'], 
                         daily_net_sentiment['Date'], 
                         frac=0.1)

# Create a DataFrame for Hillary's smoothed values
hillary_smoothed_df = pd.DataFrame({
    'Date': hillary_smoothed[:, 0],  # Keep original datetime
    'Hillary_Linear': hillary_smoothed[:, 1]
})

# Create a DataFrame for Trump's smoothed values
trump_smoothed_df = pd.DataFrame({
    'Date': trump_smoothed[:, 0],  # Keep original datetime
    'Trump_Linear': trump_smoothed[:, 1]
})

# Ensure the 'Date' column is in datetime format
hillary_smoothed_df['Date'] = pd.to_datetime(hillary_smoothed_df['Date'])
trump_smoothed_df['Date'] = pd.to_datetime(trump_smoothed_df['Date'])

# Set the 'Date' column as the index for both smoothed DataFrames
hillary_smoothed_df.set_index('Date', inplace=True)
trump_smoothed_df.set_index('Date', inplace=True)

# Optionally, reindex to match the original DataFrame's index
hillary_smoothed_df = hillary_smoothed_df.reindex(daily_net_sentiment.set_index('Date').index)
trump_smoothed_df = trump_smoothed_df.reindex(daily_net_sentiment.set_index('Date').index)

# Confidence Interval Calculation

# Merge the smoothed values back into the original DataFrame
daily_net_sentiment = daily_net_sentiment.merge(hillary_smoothed_df, on='Date', how='outer')
daily_net_sentiment = daily_net_sentiment.merge(trump_smoothed_df, on='Date', how='outer')

# Calculate standard deviation of the sentiment scores
hillary_std = np.std(daily_net_sentiment['Hillary_Net_Sentiment'])
trump_std = np.std(daily_net_sentiment['Trump_Net_Sentiment'])

# Number of observations
n = daily_net_sentiment['Date'].notnull().sum()  # Count of non-null dates

# Calculate standard errors
hillary_se = hillary_std / np.sqrt(n)
trump_se = trump_std / np.sqrt(n)

# 95% confidence level
confidence_level = 1.96  # For a 95% confidence interval

# Calculate confidence intervals for Hillary Clinton
hillary_upper = daily_net_sentiment['Hillary_Linear'] + (confidence_level * hillary_se)
hillary_lower = daily_net_sentiment['Hillary_Linear'] - (confidence_level * hillary_se)

# Calculate confidence intervals for Donald Trump
trump_upper = daily_net_sentiment['Trump_Linear'] + (confidence_level * trump_se)
trump_lower = daily_net_sentiment['Trump_Linear'] - (confidence_level * trump_se)

# ------------------- CRITICAL MOVEMENT PREPROCESSING (END) -------------------

# ------------------- CRITICAL MOVEMENT PREPROCESSING - PREDICTION (START) -------------------

# ------------------- PREPROCESSING FOR THE PREDICTION (START) -------------------

# Assume your list of themes and their corresponding titles and keywords
themes = [f'Theme_{i+1}' for i in range(13)]
titles = [
    "Campaign Invitation",
    "Hillary Clinton Foreign Policy (Border Control, Refugees Policy)",
    "Donald Trump Policy on Gun Violence and Foreign Policy",
    "Family related Policy (Gold star Family treatment + Abortion stance)",
    "Election Day",
    "Florida Campaign Rally",
    "SMS Vote support for Hillary Clinton",
    "Family Prayer Support for Candidate",
    "Policy regarding disability",
    "Positive Appreciation on Law Enforcement Performance",
    "Athletes Positive Performance during Olympic",
    "Donald Trump Plan/Policy for Working Class",
    "Trump Deportation Policy"
]

# Manually define strength relationships (placeholder values)
strength_matrix = [
    [0, 10, 12, 6, 7, 11, 8, 2, 3, 5, 9, 4, 1],  # Theme 1
    [10, 0, 11, 1, 5, 12, 8, 3, 4, 7, 9, 6, 2],  # Theme 2
    [11, 9, 0, 8, 10, 12, 7, 4, 3, 5, 6, 2, 1],  # Theme 3
    [5, 3, 10, 0, 12, 7, 8, 11, 9, 6, 4, 2, 1],  # Theme 4
    [4, 3, 9, 12, 0, 7, 10, 11, 8, 6, 5, 2, 1],  # Theme 5
    [10, 11, 12, 2, 6, 0, 9, 4, 5, 7, 8, 3, 1],  # Theme 6
    [1, 4, 5, 3, 8, 7, 0, 10, 11, 12, 9, 6, 2],  # Theme 7
    [1, 2, 5, 8, 11, 6, 10, 0, 12, 9, 7, 4, 3],  # Theme 8
    [1, 2, 3, 4, 9, 5, 11, 12, 0, 10, 8, 7, 6],  # Theme 9
    [2, 4, 3, 1, 5, 6, 11, 8, 9, 0, 12, 10, 7],  # Theme 10
    [2, 8, 3, 1, 4, 7, 10, 5, 9, 12, 0, 11, 6],  # Theme 11
    [2, 7, 3, 1, 4, 5, 9, 6, 8, 11, 12, 0, 10],  # Theme 12
    [1, 4, 3, 2, 5, 6, 8, 7, 9, 11, 10, 12, 0]   # Theme 13
]

# Transpose the matrix using zip
strength_matrix = [list(row) for row in zip(*strength_matrix)]

# Create DataFrame
themes_relationship = pd.DataFrame(strength_matrix, index=themes, columns=themes)

# Add titles as a new column
themes_relationship['Titles'] = titles

# Add titles as a new column
themes_relationship['Titles'] = titles

# Add a new column to store the corresponding themes
themes_relationship['Theme'] = themes

# Define net sentiment scores and average engagement
net_sentiment_scores = [4.7, 10.5, 4.2, 1.55, 1, 7.35, 1.7, 8.3, 0.8, 10.9, 6.6, 7.25, -1.25]
themes_avg_engagement = [14564.17, 19118.15, 17942.90, 14569.76, 12608.95, 20485.80, 5905.29, 18238.19, 15215.12, 
                         19147.81, 23350.54, 8156.73, 6697.44]
average_tweet_counts = [18, 26, 78, 29, 21, 20, 7, 21, 43, 32, 13, 26, 16]
net_sentiment_scores_avg = [net_score / tweet_count for net_score, tweet_count in zip(net_sentiment_scores, average_tweet_counts)]


# Add the columns to the DataFrame
themes_relationship['Net_Sentiment_Score'] = net_sentiment_scores
themes_relationship['Theme_Average_Engagement'] = themes_avg_engagement
themes_relationship['Average_Net_Sentiment_Score'] = net_sentiment_scores_avg

# Normalize engagement scores
scaler = MinMaxScaler()
standardized_engagement = scaler.fit_transform(themes_relationship[['Theme_Average_Engagement']])
# print(standardized_engagement.flatten())

# Adjust the net sentiment scores by multiplying with engagement
themes_relationship['Adjusted_Net_Sentiment_Score'] = themes_relationship['Average_Net_Sentiment_Score'] * standardized_engagement.flatten()

# Display the updated DataFrame
# print(themes_relationship)

# ------------------- PREPROCESSING FOR THE PREDICTION (END) -------------------

# ------------------- GENERATING PREDICTIONS (START) -------------------

# Get 'Date', 'Hillary_Net_Sentiment', 'Trump_Net_Sentiment' from daily_net_sentiment
daily_net_sentiment_pred = daily_net_sentiment[['Date', 'Hillary_Net_Sentiment', 'Trump_Net_Sentiment']].copy()
# Get an original copy of the df for later comparison 
original_daily_net_sentiment_pred = daily_net_sentiment_pred.copy()
original_daily_net_sentiment_pred['Date'] = pd.to_datetime(original_daily_net_sentiment_pred['Date'])
daily_net_sentiment_pred['Date'] = pd.to_datetime(daily_net_sentiment_pred['Date'])

# Define the date range for processing 
first_date = pd.to_datetime('2016-07-28')
start_date = pd.to_datetime('2016-07-28') 
end_date = pd.to_datetime(daily_net_sentiment_pred['Date'].max())

# Initialize Theme 04 on July 28th
theme_to_replace = 'Theme_1'
relevant_themes = []

# Iterate through each day from start date
current_date = start_date 

while current_date <= end_date:

    # First day situation
    if current_date == first_date:
        # theme decide to change for the first day
        new_theme = 'Theme_2'

        # Get the theme corresponding avg net sentiment scores 
        theme_to_replace_avg_score = themes_relationship.loc[theme_to_replace, 'Average_Net_Sentiment_Score']
        update_new_theme_avg_score = themes_relationship.loc[new_theme, 'Average_Net_Sentiment_Score']

        # Calculate the net change score
        net_change_score = update_new_theme_avg_score - theme_to_replace_avg_score

        # Update the sentiment scores
        daily_net_sentiment_pred.loc[daily_net_sentiment_pred['Date'] == start_date, 'Hillary_Net_Sentiment'] += net_change_score
        daily_net_sentiment_pred.loc[daily_net_sentiment_pred['Date'] == start_date, 'Trump_Net_Sentiment'] -= net_change_score

        # Update the current theme_to_replace 
        theme_to_replace = new_theme
        # Get the top 3 most relevant themes from themes_relationship based on the relationship scores
        top_indices = themes_relationship[theme_to_replace].nlargest(4).index.tolist()
        relevant_themes = themes_relationship.loc[top_indices, 'Theme'].tolist()
    else: 
        # Calculate the average sentiment scores for themes
        theme_to_replace_avg_score = themes_relationship.loc[theme_to_replace, 'Average_Net_Sentiment_Score']
        new_theme = relevant_themes[0]  # Select the most relevant theme for the next day
        update_new_theme_avg_score = themes_relationship.loc[new_theme, 'Average_Net_Sentiment_Score']

        # Calculate net change score
        net_change_score = update_new_theme_avg_score - theme_to_replace_avg_score

        # Update the sentiment scores for the current day
        daily_net_sentiment_pred.loc[daily_net_sentiment_pred['Date'] == current_date, 'Hillary_Net_Sentiment'] += net_change_score
        daily_net_sentiment_pred.loc[daily_net_sentiment_pred['Date'] == current_date, 'Trump_Net_Sentiment'] -= net_change_score

        # Randomly select a new theme from relevant themes that isn't the last used theme
        theme_to_replace = random.choice([t for t in relevant_themes])
        # # Update the theme_to_replace to the most relevant one for the next day
        # theme_to_replace = relevant_themes[0]  # Just select the first one from the list

        # Get the top 3 most relevant themes from themes_relationship based on the relationship scores
        top_indices = themes_relationship[theme_to_replace].nlargest(4).index.tolist()
        relevant_themes = themes_relationship.loc[top_indices, 'Theme'].tolist()

        # # Print the net change score for debugging
        # print(f"Date: {current_date}, Theme to change: {theme_to_replace}, Net Change Score: {net_change_score}")

    # Move to the next day
    current_date += pd.Timedelta(days=1)

# ------------------- GENERATING PREDICTIONS (END) -------------------

# ------------------- GENERATING LINEAR PREDICTION LINES - LOWESS (START) -------------------
# getting linear prediction graph 

daily_net_sentiment_pred['Date'] = pd.to_datetime(daily_net_sentiment_pred['Date'])

# Smooth Hillary Clinton's sentiment scores using the original dates
hillary_smoothed = lowess(daily_net_sentiment_pred['Hillary_Net_Sentiment'], 
                           daily_net_sentiment_pred['Date'], 
                           frac=0.1)

# Smooth Donald Trump's sentiment scores using the original dates
trump_smoothed = lowess(daily_net_sentiment_pred['Trump_Net_Sentiment'], 
                         daily_net_sentiment_pred['Date'], 
                         frac=0.1)

# Create a DataFrame for Hillary's smoothed values
hillary_smoothed_df = pd.DataFrame({
    'Date': hillary_smoothed[:, 0],  # Keep original datetime
    'Hillary_Linear': hillary_smoothed[:, 1]
})

# Create a DataFrame for Trump's smoothed values
trump_smoothed_df = pd.DataFrame({
    'Date': trump_smoothed[:, 0],  # Keep original datetime
    'Trump_Linear': trump_smoothed[:, 1]
})

# Ensure the 'Date' column is in datetime format
hillary_smoothed_df['Date'] = pd.to_datetime(hillary_smoothed_df['Date'])
trump_smoothed_df['Date'] = pd.to_datetime(trump_smoothed_df['Date'])

# Set the 'Date' column as the index for both smoothed DataFrames
hillary_smoothed_df.set_index('Date', inplace=True)
trump_smoothed_df.set_index('Date', inplace=True)

# Optionally, reindex to match the original DataFrame's index
hillary_smoothed_df = hillary_smoothed_df.reindex(daily_net_sentiment_pred.set_index('Date').index)
trump_smoothed_df = trump_smoothed_df.reindex(daily_net_sentiment_pred.set_index('Date').index)

# Confidence Interval Calculation

# Merge the smoothed values back into the original DataFrame
daily_net_sentiment_pred = daily_net_sentiment_pred.merge(hillary_smoothed_df, on='Date', how='outer')
daily_net_sentiment_pred = daily_net_sentiment_pred.merge(trump_smoothed_df, on='Date', how='outer')

# Save dataframe as csv file
# daily_net_sentiment_pred.to_csv('daily_net_sentiment_pred.csv', index=False, encoding='utf-8')

# ------------------- GENERATING LINEAR PREDICTION LINES - LOWESS (END) -------------------

# ------------------- CRITICAL MOVEMENT PREPROCESSING - PREDICTION (END) -------------------


# ------------------- Dash App Setup -------------------

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Campaign Insights Dashboard"

# ------------------- Dash App Layout -------------------

app.layout = dbc.Container([

    dcc.Store(id='selected-cells-store', data=[]),
    dcc.Store(id='base-distance-map-2d', data=fig_distance_map.to_json()),
    dcc.Store(id='base-distance-map-3d', data=fig_3d.to_json()),
    dcc.Store(id='selected-dates-store', data={
        'start_date': df['date'].min().isoformat(),
        'end_date': df['date'].max().isoformat()
    }),

    dbc.Row([
        dbc.Col(
            html.H1(
                "Campaign Insights Dashboard: Thematic and Sentiment Analysis of Clinton vs Trump 2016 Election Tweets",
                className='text-center mb-4'
            ),
            width=12
        )
    ]),

    dbc.Row([
        dbc.Col(
            html.H3(
                "COMP5048/4448 Assignment 2",
                className='text-center mb-4'
            ),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Label("Hansel Matthew"),
            xs=12, sm=6, md=3, lg=3, xl=3
        ),
        dbc.Col(
            dbc.Label("Tivensandro"),
            xs=12, sm=6, md=3, lg=3, xl=3
        ),
        dbc.Col(
            dbc.Label("Taige Liu"),
            xs=12, sm=6, md=3, lg=3, xl=3
        ),
        dbc.Col(
            dbc.Label("Wenhao Li"),
            xs=12, sm=6, md=3, lg=3, xl=3
        ),
    ], className='text-center mb-4'),

    dbc.Row([
        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=df['date'].min(),
                max_date_allowed=df['date'].max(),
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            )
        ], xs=12, sm=12, md=8, lg=8, xl=8),
        dbc.Col([
            dbc.Button(
                "Reset All",
                id='reset-all-button',
                color='warning',
                className='mt-4 mt-md-0',
                n_clicks=0
            )
        ], xs=12, sm=12, md=4, lg=4, xl=4, className='text-md-right text-center mt-4 mt-md-0')
    ], className='mb-4', align='center'),
    
    # Distance Maps Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("2D Distance Map (U-Matrix)"),
                dbc.CardBody([
                    dcc.Graph(
                        id='distance-map-2d',
                        figure=fig_distance_map,
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '1200px'}  # Increased height from 1000px to 1200px
                    )
                ], style={'padding': '0'})
            ], className='h-100')
        ], xs=12, sm=12, md=12, lg=6, xl=6, className='mb-4'),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("3D Engagement Map"),
                dbc.CardBody([
                    dcc.Graph(
                        id='distance-map-3d',
                        figure=fig_3d,
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '600px'}
                    )
                ], style={'padding': '0'})
            ], className='h-100')
        ], xs=12, sm=12, md=12, lg=6, xl=6, className='mb-4'),
    ], className='mb-4'),
    
    # Visualization Plots: t-SNE and UMAP
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("t-SNE Visualization (Log Engagement)"),
                dbc.CardBody([
                    dcc.Graph(
                        id='tsne-plot',
                        figure=px.scatter(
                            vis_df_tsne, x='x', y='y',
                            color='engagement_log',
                            color_continuous_scale='Viridis',
                            hover_data=['tweet', 'engagement'],
                            labels={'engagement_log': 'Engagement (Log)'},
                            title='',
                            width=600,
                            height=600
                        ),
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '100%'}
                    )
                ], style={'padding': '0'})
            ], className='h-100 mb-4')
        ], xs=12, sm=12, md=6, lg=6, xl=6, className='mb-4'),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("UMAP Visualization (Log Engagement)"),
                dbc.CardBody([
                    dcc.Graph(
                        id='umap-plot',
                        figure=px.scatter(
                            vis_df_umap, x='x', y='y',
                            color='engagement_log',
                            color_continuous_scale='Viridis',
                            hover_data=['tweet', 'engagement'],
                            labels={'engagement_log': 'Engagement (Log)'},
                            title='',
                            width=600,
                            height=600
                        ),
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '100%'}
                    )
                ], style={'padding': '0'})
            ], className='h-100 mb-4')
        ], xs=12, sm=12, md=6, lg=6, xl=6, className='mb-4'),
    ], className='mb-4'),
    
    # Word Frequency Bar Chart Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Word Frequency in Selected Tweets"),
                dbc.CardBody([
                    dcc.Graph(
                        id='word-frequency-chart',
                        figure={},  # Initially empty
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '400px'}
                    )
                ], style={'padding': '0'})
            ], className='h-100 mb-4')
        ], xs=12, sm=12, md=12, lg=12, xl=12, className='mb-4')
    ], className='mb-4'),
            
    # Sentiment Scatter Plot Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Scatter Plot"),
                dbc.CardBody([
                    dcc.Graph(
                        id='sentiment-scatter-plot',
                        figure={},  # Initially empty; will be updated via callback
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '500px'}
                    )
                ], style={'padding': '0'})
            ], className='h-100 mb-4')
        ], xs=12, sm=12, md=12, lg=12, xl=12, className='mb-4')
    ], className='mb-4'),
    
    # Line Charts Row (Merged into one)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Average Polarity Over Time"),
                dbc.CardBody([
                    dcc.Graph(
                        id='polarity-line-chart',
                        figure={},  # Initially empty; will be updated via callback
                        config={'displayModeBar': False},
                        style={'width': '100%', 'height': '500px'}
                    )
                ])
            ]),
        ]),

    ]),

    # Tweets Table Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tweets in Selected Cells"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='tweets-table',
                        columns=[
                            {"name": "Tweet", "id": "tweet"},
                            {"name": "Engagement", "id": "engagement"},
                            {"name": "Log Engagement", "id": "engagement_log"},
                            {"name": "Sentiment Score", "id": "Sentiment Score"}  # New Column
                        ],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '5px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'padding': '0'})
            ], className='h-100 mb-4')
        ], width=12)
    ]),

    # Selected Cell Information Row
    dbc.Row([
        dbc.Col(
            html.Div(
                id='selected-cell',
                style={'textAlign': 'center', 'fontSize': '1.25rem', 'marginTop': '20px'}
            ),
            width=12
        )
    ], className='mb-4'),

    # ------------------------- CRITICAL MOVEMENT (START) -------------------------
    dbc.Row([
        dbc.Col(dcc.Graph(id='daily-net-sentiment-plot'), width=12),  
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='net_sentiment_plot_linear'), width=12),  
    ]),

    html.Div(id='hover-data', style={'margin-top': '20px'}),  # This can be for displaying clicked point info
    dcc.Graph(id='info-table'),  # New graph for the data table

    # ------------------------- CRITICAL MOVEMENT (END) -------------------------

    # ------------------------- CRITICAL MOVEMENT - PREDICTION (START) -------------------------

    dbc.Row([
        dbc.Col(dcc.Graph(id='daily-net-sentiment-plot-pred'), width=12),  
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='net_sentiment_plot_linear_pred'), width=12),  
    ]),

    html.Div(id='hover-data-pred', style={'margin-top': '20px'}),  # This can be for displaying clicked point info
    dcc.Graph(id='info-table-pred'),  # New graph for the data table

    # ------------------------- CRITICAL MOVEMENT - PREDICTION (END) -------------------------

    
], fluid=True)



# ------------------- Callbacks for Interactivity -------------------

@app.callback(
    [
        Output('tsne-plot', 'figure'),
        Output('umap-plot', 'figure'),
        Output('selected-cell', 'children'),
        Output('tweets-table', 'data'),
        Output('distance-map-2d', 'figure'),
        Output('distance-map-3d', 'figure'),
        Output('word-frequency-chart', 'figure'),
        Output('sentiment-scatter-plot', 'figure'),
        Output('polarity-line-chart', 'figure'),
        Output('selected-dates-store', 'data')
    ],
    [
        Input('selected-cells-store', 'data'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ],
    [
        State('base-distance-map-2d', 'data'),
        State('base-distance-map-3d', 'data'),
        State('selected-dates-store', 'data')
    ]
)
def update_visualizations(selected_cells, start_date, end_date, base_distance_map_2d_json, base_distance_map_3d_json, selected_dates_store):
    """
    Updates the visualizations and tables based on the list of selected cells and selected date range.
    """
    # Load base distance maps from JSON
    base_distance_map_2d = pio.from_json(base_distance_map_2d_json)
    base_distance_map_3d = pio.from_json(base_distance_map_3d_json)
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Convert start_date and end_date to datetime.date objects
    if start_date is not None:
        start_date = pd.to_datetime(start_date).date()
    else:
        start_date = df['date'].min()
    if end_date is not None:
        end_date = pd.to_datetime(end_date).date()
    else:
        end_date = df['date'].max()
    
    # Update selected dates store
    selected_dates_store = {'start_date': start_date.isoformat(), 'end_date': end_date.isoformat()}
    
    # Filter data based on date range without resetting the index
    date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_filtered = df[date_mask]
    sentiment_df_filtered = sentiment_df[date_mask]
    vis_df_tsne_filtered = vis_df_tsne[date_mask]
    vis_df_umap_filtered = vis_df_umap[date_mask]
    vis_df_som_filtered = vis_df_som[date_mask]
    
    # Determine whether filters are at default (full date range and no SOM selection)
    date_min = df['date'].min()
    date_max = df['date'].max()
    start_date_default = start_date == date_min
    end_date_default = end_date == date_max
    filters_at_default = start_date_default and end_date_default and not selected_cells

    # Determine selected indices based on selected cells
    if selected_cells:
        # Get indices of tweets in selected cells
        selected_indices = [index for cell in selected_cells for index in grid_to_tweets.get(tuple(cell), [])]
        selected_indices = list(set(selected_indices))
        # Apply date filter to selected indices
        selected_indices = [i for i in selected_indices if date_mask.loc[i]]
    else:
        # If no cells are selected, use all indices in the date range
        selected_indices = df_filtered.index.tolist()

    # Filter sentiment data for selected tweets
    sentiment_selected = sentiment_df_filtered.loc[selected_indices]

    # ------------------- **Add Tweet Text to sentiment_selected** -------------------
    # This allows the tweet text to be used in the hover tooltip of the scatter plot
    sentiment_selected = sentiment_selected.copy()  # To avoid SettingWithCopyWarning
    sentiment_selected['tweet'] = df_filtered.loc[selected_indices, 'text'].values

    # Update tweets data for the table
    if filters_at_default:
        # No filters applied; do not show tweets
        tweets_data = []
        selected_cell_text = html.Div([
            html.P("No tweets selected. Please select a date range or SOM cells.")
        ])
    else:
        # ------------------- **Add Sentiment Score to the Tweets Table** -------------------
        # Merge the sentiment score into the tweets data
        tweets_data_df = df_filtered.loc[selected_indices, ['text', 'engagement', 'engagement_log']].copy()
        tweets_data_df.rename(columns={'text': 'tweet'}, inplace=True)
        tweets_data_df['Sentiment Score'] = sentiment_selected['Sentiment Score'].values  # Add Sentiment Score

        tweets_data = tweets_data_df.to_dict('records')
        
        # Aggregate selected cell information
        if selected_cells:
            total_engagement = df_filtered.loc[selected_indices, 'engagement'].sum()
            average_engagement = df_filtered.loc[selected_indices, 'engagement'].mean()
            selected_cell_text = html.Div([
                html.P(f"Selected SOM Cells: {len(selected_cells)}"),
                html.P(f"Cells: {selected_cells}"),
                html.P(f"Number of Tweets: {len(selected_indices)}"),
                html.P(f"Total Engagement: {int(total_engagement)}"),
                html.P(f"Average Engagement: {average_engagement:.2f}")
            ])
        else:
            selected_cell_text = html.Div([
                html.P(f"Number of Tweets in Date Range: {len(df_filtered)}")
            ])

    # Handle case when no data is available after filtering
    if df_filtered.empty or not selected_indices:
        # Create empty plots with a message
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=20)
        )
        return empty_fig, empty_fig, selected_cell_text, tweets_data, \
               base_distance_map_2d, base_distance_map_3d, empty_fig, empty_fig, empty_fig, selected_dates_store

    # ------------------- **Update Sentiment Scatter Plot with Color Blind Friendly Colors** -------------------
    sentiment_plot = px.scatter(
        sentiment_selected,
        x='Sentiment Score',
        y='Engagement',
        color='Handle',
        title='Sentiment Polarity vs Total Engagement (Filtered Data)',
        labels={'Sentiment Score': 'Sentiment Polarity', 'Engagement': 'Total Engagement'},
        hover_data={'tweet': True},  # Enable tweet text in hover
        color_discrete_sequence=px.colors.qualitative.Safe  # Color blind friendly palette
    )
    sentiment_plot.update_traces(marker=dict(size=10))
    sentiment_plot.update_layout(template="plotly_white")

    # ------------------- **Update Polarity Line Chart with Color Blind Friendly Colors** -------------------
    if not sentiment_selected.empty:
        # Ensure 'Date' column has valid dates
        if sentiment_selected['Date'].isna().any():
            sentiment_selected = sentiment_selected.dropna(subset=['Date'])

        # Group by Date and Handle for per-handle sentiment
        sentiment_time_series = sentiment_selected.groupby(['Date', 'Handle'])['Sentiment Score'].mean().unstack()
        sentiment_time_series = sentiment_time_series.replace(0, np.nan)

        # Initialize the figure
        fig_line = go.Figure()

        # Define color sequence
        color_sequence = px.colors.qualitative.Safe
        num_colors = len(color_sequence)

        # Add per-handle sentiment lines
        for i, handle in enumerate(sentiment_time_series.columns):
            fig_line.add_trace(
                go.Scatter(
                    x=sentiment_time_series.index,  # Dates
                    y=sentiment_time_series[handle],  # Average sentiment score for each handle
                    mode='lines+markers',  # Add markers to make single data points visible
                    name=f"{handle} Sentiment",
                    line=dict(color=color_sequence[i % num_colors])  # Assign color from Safe palette
                )
            )

        # Calculate overall sentiment
        sentiment_overall_time_series = sentiment_selected.groupby('Date')['Sentiment Score'].mean()

        # Add overall sentiment line
        fig_line.add_trace(
            go.Scatter(
                x=sentiment_overall_time_series.index,
                y=sentiment_overall_time_series.values,
                mode='lines+markers',
                name='Overall Sentiment',
                line=dict(color=color_sequence[-1], width=4, dash='dash')  # Use the last color for distinction
            )
        )

        # Update the layout of the combined figure
        fig_line.update_layout(
            title='Average Sentiment Analysis Over Time (By Handle and Overall)',
            xaxis_title='Date',
            yaxis_title='Average Sentiment Score',
            legend_title='Handles',
            hovermode='x unified',
            template='plotly_white'
        )

    else:
        # If no data, create empty figure with a message
        fig_line = go.Figure()
        fig_line.add_annotation(
            text="No data available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=20)
        )
        fig_line.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

    # ------------------- **Update t-SNE Plot** -------------------
    mask_tsne = vis_df_tsne_filtered.index.isin(selected_indices)
    if selected_cells:
        # Only plot the selected tweets
        data_to_plot_tsne = vis_df_tsne_filtered.loc[mask_tsne]
    else:
        # Plot all tweets
        data_to_plot_tsne = vis_df_tsne_filtered

    fig_tsne = px.scatter(
        data_to_plot_tsne,
        x='x',
        y='y',
        color='engagement_log',
        color_continuous_scale='Viridis',
        hover_data=['tweet', 'engagement'],
        labels={'engagement_log': 'Engagement (Log)'},
        title='t-SNE Visualization (Log Engagement)',
        width=600,
        height=600
    )

    # ------------------- **Update UMAP Plot** -------------------
    mask_umap = vis_df_umap_filtered.index.isin(selected_indices)
    if selected_cells:
        # Only plot the selected tweets
        data_to_plot_umap = vis_df_umap_filtered.loc[mask_umap]
    else:
        # Plot all tweets
        data_to_plot_umap = vis_df_umap_filtered

    fig_umap = px.scatter(
        data_to_plot_umap,
        x='x',
        y='y',
        color='engagement_log',
        color_continuous_scale='Viridis',
        hover_data=['tweet', 'engagement'],
        labels={'engagement_log': 'Engagement (Log)'},
        title='UMAP Visualization (Log Engagement)',
        width=600,
        height=600
    )

    # Update distance-map-2d with selected cells highlighted in black
    fig_distance_map_2d = go.Figure(base_distance_map_2d)  # Make a copy
    if selected_cells:
        selected_x = [cell[0] for cell in selected_cells]
        selected_y = [cell[1] for cell in selected_cells]
        fig_distance_map_2d.add_trace(
            go.Scatter(
                x=selected_x,
                y=selected_y,
                mode='markers',
                marker=dict(
                    color='black',
                    size=10,
                    symbol='square',
                    line=dict(width=2, color='white')
                ),
                name='Selected Cells',
                hoverinfo='text',
                text=[f"SOM X: {x}, SOM Y: {y}" for x, y in zip(selected_x, selected_y)]
            )
        )

    # Update distance-map-3d with selected cells highlighted in black
    fig_distance_map_3d = go.Figure(base_distance_map_3d)  # Make a copy
    if selected_cells:
        selected_x_3d = [x for x, y in selected_cells]
        selected_y_3d = [y for x, y in selected_cells]
        selected_z = [z_data[y, x] for x, y in selected_cells]  # Use original engagement values for z-coordinate
        fig_distance_map_3d.add_trace(
            go.Scatter3d(
                x=selected_x_3d,
                y=selected_y_3d,
                z=selected_z,
                mode='markers',
                marker=dict(
                    color='black',
                    size=5,
                    symbol='circle'
                ),
                name='Selected Cells',
                hoverinfo='text',
                text=[f"SOM X: {x}, SOM Y: {y}, Engagement: {z:.2f}"
                      for x, y, z in zip(selected_x_3d, selected_y_3d, selected_z)]
            )
        )

    # ------------------- Generate Horizontal Word Frequency Bar Chart -------------------

    if not filters_at_default and not tweets_data_df.empty:
        # Get the cleaned tweets corresponding to selected indices
        tweets_cleaned_selected = tweets_cleaned.loc[selected_indices]

        # Extract all words from selected cleaned tweets
        all_words = ' '.join(tweets_cleaned_selected).split()

        # Count word frequencies
        word_counts = Counter(all_words)

        # Get the top 40 words
        top_words = word_counts.most_common(40)

        if top_words:
            words, counts = zip(*top_words)
            # Create the horizontal bar chart
            fig_word_freq = px.bar(
                y=words,            # Assign words to the y-axis for horizontal bars
                x=counts,           # Assign counts to the x-axis
                labels={'x': 'Frequency', 'y': 'Words'},  # Update labels accordingly
                title='Top 40 Word Frequencies in Selected Tweets',
                color=counts,
                color_continuous_scale='Viridis',
                orientation='h'     # Set orientation to horizontal
            )
            fig_word_freq.update_layout(
                yaxis={'categoryorder': 'total ascending'},  # Order bars from least to most frequent
                plot_bgcolor='white',
                margin=dict(l=150, r=50, t=50, b=50)      # Adjust left margin for labels
            )
        else:
            # If no words are present
            fig_word_freq = go.Figure()
            fig_word_freq.add_annotation(
                dict(
                    text="No words to display.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=20)
                )
            )
            fig_word_freq.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
    else:
        # If no tweets are selected, display an empty chart with a message
        fig_word_freq = go.Figure()
        fig_word_freq.add_annotation(
            dict(
                text="No data available.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=20)
            )
        )
        fig_word_freq.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

    # ------------------- **Return All Updated Figures and Data** -------------------
    return fig_tsne, fig_umap, selected_cell_text, tweets_data, \
           fig_distance_map_2d, fig_distance_map_3d, fig_word_freq, sentiment_plot, fig_line, selected_dates_store


@app.callback(
    [
        Output('selected-cells-store', 'data'),
        Output('date-picker-range', 'start_date'),
        Output('date-picker-range', 'end_date')
    ],
    [
        Input('distance-map-2d', 'clickData'),
        Input('distance-map-3d', 'clickData'),
        Input('reset-all-button', 'n_clicks'),
        # Added Inputs for Date Picker to handle two-way interaction
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ],
    [
        State('selected-cells-store', 'data'),
        State('selected-dates-store', 'data')
    ]
)
def update_selected_cells_and_date(clickData_2d, clickData_3d, reset_all_clicks, start_date, end_date, selected_cells, selected_dates_store):
    """
    Updates the list of selected cells and the date picker based on user interactions.
    Handles clearing selections and resetting the date filter.
    Also handles updating selected cells based on date range changes.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default start and end dates
    current_start_date = pd.to_datetime(selected_dates_store['start_date']).date()
    current_end_date = pd.to_datetime(selected_dates_store['end_date']).date()

    if triggered_id == 'reset-all-button':
        # Reset selected cells
        selected_cells = []
        # Reset date picker to full range
        start_date = df['date'].min()
        end_date = df['date'].max()
        return selected_cells, start_date, end_date

    elif triggered_id in ['distance-map-2d', 'distance-map-3d']:
        # Determine which distance map was clicked
        if triggered_id == 'distance-map-2d':
            clickData = clickData_2d
        else:
            clickData = clickData_3d

        if clickData is None:
            raise PreventUpdate

        # Extract grid cell coordinates from clickData
        point = clickData['points'][0]
        som_x = int(point['x'])
        som_y = int(point['y'])

        clicked_cell = (som_x, som_y)

        if clicked_cell in selected_cells:
            # If already selected, remove it (toggle off)
            selected_cells.remove(clicked_cell)
        else:
            # Add to selected cells
            selected_cells.append(clicked_cell)

        # Get dates from data points in selected cells
        indices_in_selected_cells = [index for cell in selected_cells for index in grid_to_tweets.get(tuple(cell), [])]
        dates_in_selected_cells = df.loc[indices_in_selected_cells, 'date']

        if not dates_in_selected_cells.empty:
            start_date = dates_in_selected_cells.min()
            end_date = dates_in_selected_cells.max()
        else:
            # If no dates found, reset to full range
            start_date = df['date'].min()
            end_date = df['date'].max()

        return selected_cells, start_date, end_date

    elif triggered_id in ['date-picker-range']:
        # When date picker changes, update selected cells based on the new date range

        # Convert to datetime.date objects
        if start_date is not None:
            start = pd.to_datetime(start_date).date()
        else:
            start = df['date'].min()
        if end_date is not None:
            end = pd.to_datetime(end_date).date()
        else:
            end = df['date'].max()

        # Use vis_df_som to access 'som_x' and 'som_y'
        filtered_tweets = vis_df_som[(vis_df_som['date'] >= start) & (vis_df_som['date'] <= end)]

        # Extract unique SOM cell coordinates
        relevant_cells = filtered_tweets[['som_x', 'som_y']].drop_duplicates().values.tolist()

        # Update selected_cells to be the relevant_cells
        selected_cells = relevant_cells

        return selected_cells, start_date, end_date

    else:
        # If no recognized input triggered the callback, prevent update
        raise PreventUpdate


# ------------------- Critical Movement Graphs (START) -------------------

@app.callback(
    [
        Output('daily-net-sentiment-plot', 'figure'), 
        Output('net_sentiment_plot_linear', 'figure')
    ],
    [
        Input('net_sentiment_plot_linear', 'clickData'),
    ]
)

# ------------------- Generate Sentiment Over Date Graph-------------------
def update_net_sentiment_plot(clickData):  

    # Create the scatter plot
    net_sentiment_plot = go.Figure()

    # Plot Hillary Clinton's normalized sentiment
    net_sentiment_plot.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Hillary_Net_Sentiment'],
        mode='lines+markers',
        name='Hillary Clinton Net Sentiment',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Plot Donald Trump's normalized sentiment
    net_sentiment_plot.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Trump_Net_Sentiment'],
        mode='lines+markers',
        name='Donald Trump Net Sentiment',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))

    # Optional: Add a horizontal line at y=0 for reference
    net_sentiment_plot.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=[0]*len(daily_net_sentiment),  # Line at y=0
        mode='lines',
        name='Neutral Sentiment',
        line=dict(color='grey', dash='dash'),
    ))

    # Update the layout for the plot
    net_sentiment_plot.update_layout(
        title='Daily Net Sentiment Score for Hillary Clinton and Donald Trump',
        xaxis_title='Date',
        yaxis_title='Net Sentiment Score',
        yaxis=dict(range=[-1, 1]),  # Adjust range based on your data
        template='plotly_white'
    )

    # ------------------- Generate Sentiment Over Date Linear Graph -------------------

    # Create the figure
    net_sentiment_plot_linear = go.Figure()

    # Plot Hillary Clinton's smoothed line and confidence intervals
    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Hillary_Linear'],
        mode='lines',
        name='Hillary Clinton (LOWESS)',
        line=dict(color='blue'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=hillary_upper,
        mode='lines',
        name='Hillary Confidence Upper',
        line=dict(width=0),
        showlegend=False
    ))

    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=hillary_lower,
        mode='lines',
        fill='tonexty',  # Fill to next y
        fillcolor='rgba(0, 0, 255, 0.2)',  # Fill color for confidence interval
        line=dict(width=0),
        showlegend=False, 
        name='Hillary Clinton Confidence Interval'
    ))

    # Plot Donald Trump's smoothed line and confidence intervals
    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Trump_Linear'],
        mode='lines',
        name='Donald Trump (LOWESS)',
        line=dict(color='red'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=trump_upper,
        mode='lines',
        name='Trump Confidence Upper',
        line=dict(width=0),
        showlegend=False
    ))

    net_sentiment_plot_linear.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=trump_lower,
        mode='lines',
        fill='tonexty',  # Fill to next y
        fillcolor='rgba(255, 0, 0, 0.2)',  # Fill color for confidence interval
        line=dict(width=0),
        showlegend=False, 
        name='Trump Confidence Confidence Interval'
    ))

    # Update layout
    net_sentiment_plot_linear.update_layout(
        title='Smoothed Sentiment Analysis of Tweets Over Time (Clinton vs Trump)',
        xaxis_title='Date',
        yaxis_title='Normalized Net Sentiment Score',
        template='plotly_white'
    )

    # Add marker for the clicked point if it exists
    if clickData:
        x, y = (clickData['points'][0]['x'], clickData['points'][0]['y'])

        # Add marker for the clicked point
        net_sentiment_plot_linear.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(color='green', size=10, symbol='cross'),
            name='Clicked Point',
            text=[f'Clicked Point: {x}'],
            textposition='top center'
        ))

        # Add a vertical dashed line at the clicked point
        net_sentiment_plot_linear.add_shape(
            type='line',
            x0=x, y0=0, 
            x1=x, y1=y,
            line=dict(color='green', width=2, dash='dash'
        ))  

    return net_sentiment_plot, net_sentiment_plot_linear    

@app.callback(
    Output('info-table', 'figure'),
    Input('net_sentiment_plot_linear', 'clickData')
)
def update_table(clickData):

    # Create table data if clicked point exists
    if clickData:
        x, y = (clickData['points'][0]['x'], clickData['points'][0]['y'])
        date_clicked = x  # Assuming x is the date clicked
        clicked_data = daily_net_sentiment[daily_net_sentiment['Date'] == date_clicked]  
        if not clicked_data.empty:
            data = [[f'Clicked Point', date_clicked, clicked_data['Hillary_Net_Sentiment'].values[0], clicked_data['Trump_Net_Sentiment'].values[0]]]
            return go.Figure(data=[go.Table(
                header=dict(values=["Point", "Date", "Hillary Clinton Sentiment", "Donald Trump Sentiment"]),
                cells=dict(values=list(zip(*data)))
            )])
    
    return go.Figure(data=[go.Table(
        header=dict(values=["Point", "Date", "Hillary Clinton Sentiment", "Donald Trump Sentiment"]),
        cells=dict(values=[[], [], [], []]))])  # Return an empty table


# ------------------- Critical Movement Graphs (END) -------------------

# ------------------- Critical Movement Graphs - Predictions (START) -------------------


@app.callback(
    [
        Output('daily-net-sentiment-plot-pred', 'figure'), 
        Output('net_sentiment_plot_linear_pred', 'figure')
    ],
    [
        Input('net_sentiment_plot_linear_pred', 'clickData'),
    ]
)

# ------------------- Generate Predicted Sentiment Over Date -------------------
def update_net_sentiment_predicted_plot(clickData):  

    # Create the scatter plot
    net_sentiment_plot_pred = go.Figure()

    # Plot Hillary Clinton's normalized sentiment
    net_sentiment_plot_pred.add_trace(go.Scatter(
        x=daily_net_sentiment_pred['Date'],
        y=daily_net_sentiment_pred['Hillary_Net_Sentiment'],
        mode='lines+markers',
        name='Hillary Clinton Net Sentiment',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Plot Donald Trump's normalized sentiment
    net_sentiment_plot_pred.add_trace(go.Scatter(
        x=daily_net_sentiment_pred['Date'],
        y=daily_net_sentiment_pred['Trump_Net_Sentiment'],
        mode='lines+markers',
        name='Donald Trump Net Sentiment',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))

    # Optional: Add a horizontal line at y=0 for reference
    net_sentiment_plot_pred.add_trace(go.Scatter(
        x=daily_net_sentiment_pred['Date'],
        y=[0]*len(daily_net_sentiment_pred),  # Line at y=0
        mode='lines',
        name='Neutral Sentiment',
        line=dict(color='grey', dash='dash'),
    ))

    # Update the layout for the plot
    net_sentiment_plot_pred.update_layout(
        title='Daily Net Sentiment Score for Hillary Clinton and Donald Trump',
        xaxis_title='Date',
        yaxis_title='Net Sentiment Score',
        yaxis=dict(range=[-1, 1]),  # Adjust range based on your data
        template='plotly_white'
    )

    # ------------------- Generate Predicted Sentiment Over Date Linear Graph -------------------

    # Create the figure
    net_sentiment_plot_linear_pred = go.Figure()

    # Plot Hillary Clinton's original smoothed line 
    net_sentiment_plot_linear_pred.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Hillary_Linear'],
        mode='lines',
        name='Hillary Clinton (LOWESS)',
        line=dict(color='blue'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    # Plot Hillary Clinton's predicted smoothed line 
    net_sentiment_plot_linear_pred.add_trace(go.Scatter(
        x=daily_net_sentiment_pred['Date'],
        y=daily_net_sentiment_pred['Hillary_Linear'],
        mode='lines',
        name='Hillary Clinton (LOWESS)',
        line=dict(color='blue', dash='dash'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    # Plot Donald Trump's original smoothed line
    net_sentiment_plot_linear_pred.add_trace(go.Scatter(
        x=daily_net_sentiment['Date'],
        y=daily_net_sentiment['Trump_Linear'],
        mode='lines',
        name='Donald Trump (LOWESS)',
        line=dict(color='red'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    # Plot Donald Trump's predicted smoothed line 
    net_sentiment_plot_linear_pred.add_trace(go.Scatter(
        x=daily_net_sentiment_pred['Date'],
        y=daily_net_sentiment_pred['Trump_Linear'],
        mode='lines',
        name='Donald Trump (LOWESS)',
        line=dict(color='red', dash='dash'), 
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.2f}<extra></extra>'  # Customize the hover template
    ))

    # Update layout
    net_sentiment_plot_linear_pred.update_layout(
        title='Predicted Smoothed Sentiment Analysis of Tweets Over Time (Clinton vs Trump)',
        xaxis_title='Date',
        yaxis_title='Normalized Net Sentiment Score',
        template='plotly_white'
    )

    # Add marker for the clicked point if it exists
    if clickData:
        x, y = (clickData['points'][0]['x'], clickData['points'][0]['y'])

        # Add marker for the clicked point
        net_sentiment_plot_linear_pred.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(color='green', size=10, symbol='cross'),
            name='Clicked Point',
            text=[f'Clicked Point: {x}'],
            textposition='top center'
        ))

        # Add a vertical dashed line at the clicked point
        net_sentiment_plot_linear_pred.add_shape(
            type='line',
            x0=x, y0=0, 
            x1=x, y1=y,
            line=dict(color='green', width=2, dash='dash'
        ))  

    return net_sentiment_plot_pred, net_sentiment_plot_linear_pred    

# ------------------- Generate Table for displaying clicked data -------------------
@app.callback(
    Output('info-table-pred', 'figure'),
    Input('net_sentiment_plot_linear_pred', 'clickData')
)
def update_table(clickData):
    # Create table data if clicked point exists
    if clickData:
        x, y = (clickData['points'][0]['x'], clickData['points'][0]['y'])
        date_clicked = x  # Assuming x is the date clicked
        
        # Locate the row corresponding to the clicked date
        clicked_data = daily_net_sentiment_pred[daily_net_sentiment_pred['Date'] == date_clicked]  
        if not clicked_data.empty:
            # Extract necessary values
            current_hillary_sentiment = clicked_data['Hillary_Net_Sentiment'].values[0]  # Original sentiment
            predicted_hillary_sentiment = clicked_data['Hillary_Linear'].values[0]  # Predicted sentiment

            current_trump_sentiment = clicked_data['Trump_Net_Sentiment'].values[0]  # Original sentiment
            predicted_trump_sentiment = clicked_data['Trump_Linear'].values[0]  # Predicted sentiment

            data = [[
                'Clicked Point', 
                date_clicked, 
                current_hillary_sentiment, 
                predicted_hillary_sentiment,
                current_trump_sentiment,
                predicted_trump_sentiment
            ]]
            return go.Figure(data=[go.Table(
                header=dict(values=["Point", "Date", "Original Hillary Sentiment", "Predicted Hillary Sentiment", 
                                    "Original Trump Sentiment", "Predicted Trump Sentiment"]),
                cells=dict(values=list(zip(*data)))
            )])
    
    return go.Figure(data=[go.Table(
        header=dict(values=["Point", "Date", "Original Hillary Sentiment", "Predicted Hillary Sentiment", 
                            "Original Trump Sentiment", "Predicted Trump Sentiment"]),
        cells=dict(values=[[], [], [], [], [], []]))])  # Return an empty table


# ------------------- Critical Movement Graphs - Predictions (END) -------------------



# ------------------- Running the Dash App -------------------

if __name__ == '__main__':
    app.run_server(debug=True)
