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
import dash_table
from dash.exceptions import PreventUpdate
import plotly.io as pio  # For figure serialization/deserialization
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

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

# ------------------- Create 3D U-Matrix Heatmap -------------------

# Convert distance_map_df to NumPy array
z_data = 1 - distance_map_df.values
z_data_original = distance_map_df.values

# Create the 3D surface plot
surface = go.Surface(
    z=z_data,
    x=list(range(z_data.shape[1])),  # SOM X-axis
    y=list(range(z_data.shape[0])),  # SOM Y-axis
    colorscale='YlOrRd_r',  # Choose a perceptually uniform colorscale
    colorbar=dict(title='Distance'),
    hovertemplate='SOM X: %{x}<br>SOM Y: %{y}<br>Distance: %{customdata:.4f}<extra></extra>',
    customdata=z_data_original
)

# Define the layout for the 3D plot
layout_3d = go.Layout(
    title='SOM Distance Map (U-Matrix) - 3D Visualization',
    scene=dict(
        xaxis_title='SOM X',
        yaxis_title='SOM Y',
        zaxis_title='Distance',
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

# ------------------- Dash App Setup -------------------

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Interactive Tweet Visualization"

# ------------------- Dash App Layout -------------------

app.layout = dbc.Container([
    # Hidden Stores to keep track of selected cells, dates, and base distance maps
    dcc.Store(id='selected-cells-store', data=[]),
    dcc.Store(id='base-distance-map-2d', data=fig_distance_map.to_json()),
    dcc.Store(id='base-distance-map-3d', data=fig_3d.to_json()),
    dcc.Store(id='selected-dates-store', data={'start_date': df['date'].min().isoformat(),
                                               'end_date': df['date'].max().isoformat()}),
    
    # Title Row
    dbc.Row([
        dbc.Col(html.H1("Interactive Tweet Visualization with SOM, t-SNE, UMAP, and 3D U-Matrix", className='text-center mb-4'), width=12)
    ]),
    
    # Date Picker for filtering by date
    dbc.Row([
        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=df['date'].min(),
                max_date_allowed=df['date'].max(),
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                display_format='YYYY-MM-DD'
            ),
            dbc.Button(
                "Reset Date Filter",
                id='reset-date-filter-button',
                color='secondary',
                className='ml-2',
                n_clicks=0
            )
        ], width=12, className='mb-4')
    ]),
    
    # Distance Maps and Clear Selection Button
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='distance-map-2d',
                figure=fig_distance_map,
                config={'displayModeBar': False},
                style={'height': '1000px'}
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='distance-map-3d',
                figure=fig_3d,
                config={'displayModeBar': False},
                style={'height': '600px'}
            )
        ], width=6),
    ]),
    
    # Clear Selection Button
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Clear SOM Selection",
                id='clear-selection-button',
                color='danger',
                className='mt-3',
                n_clicks=0
            )
        ], width=12, className='text-center')
    ]),
    
    # Visualization Plots
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
                            title='',
                            width=600,
                            height=600
                        ),
                        config={'displayModeBar': False},
                        style={'height': '600px'}
                    )
                ])
            ], style={'height': '620px'})
        ], width=3),
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
                            title='',
                            width=600,
                            height=600
                        ),
                        config={'displayModeBar': False},
                        style={'height': '600px'}
                    )
                ])
            ], style={'height': '620px'})
        ], width=3),
    ]),
    
    # Horizontal Word Frequency Bar Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Word Frequency in Selected Tweets"),
                dbc.CardBody([
                    dcc.Graph(
                        id='word-frequency-chart',
                        figure={},  # Initially empty
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style={'height': '420px'})
        ], width=12)
    ], className='mb-4'),
    
    # Engagement Box Plot
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Engagement Distribution of Selected Tweets"),
                dbc.CardBody([
                    dcc.Graph(
                        id='engagement-boxplot',
                        figure={},  # Initially empty
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style={'height': '420px'})
        ], width=12)
    ], className='mb-4'),
    
    # Selected Cell Information
    dbc.Row([
        dbc.Col(html.Div(id='selected-cell', style={'textAlign': 'center', 'fontSize': 20, 'marginTop': '20px'}), width=12)
    ]),
    
    # Tweets Table
    dbc.Row([
        dbc.Col([
            html.H4("Tweets in Selected Cells"),
            dash_table.DataTable(
                id='tweets-table',
                columns=[
                    {"name": "Tweet", "id": "tweet"},
                    {"name": "Engagement", "id": "engagement"},
                    {"name": "Log Engagement", "id": "engagement_log"}
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
        ], width=12)
    ]),
    
    # Sentiment Scatter Plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='sentiment-scatter-plot',
                figure={},  # Initially empty; will be updated via callback
                config={'displayModeBar': False},
                style={'height': '500px'}
            )
        ], width=12),
    ]),
    
    # Line Chart for Average Polarity over Time by Handle
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='polarity-line-chart',
                figure={},  # Initially empty; will be updated via callback
                config={'displayModeBar': False},
                style={'height': '500px'}
            )
        ], width=12),
    ]),
    
    # Line Chart for Average Polarity over Time (Overall)
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='polarity-line-chart-overall',
                figure={},  # Initially empty; will be updated via callback
                config={'displayModeBar': False},
                style={'height': '500px'}
            )
        ], width=12),
    ]),
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
        Output('engagement-boxplot', 'figure'),
        Output('sentiment-scatter-plot', 'figure'),
        Output('polarity-line-chart', 'figure'),
        Output('polarity-line-chart-overall', 'figure'),
        Output('selected-dates-store', 'data')  # Store to update selected dates
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
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
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
    
    # -------------------- Adjustment Starts Here --------------------

    # Determine whether filters are at default (full date range and no SOM selection)
    date_min = df['date'].min()
    date_max = df['date'].max()
    start_date_default = start_date == date_min
    end_date_default = end_date == date_max
    filters_at_default = start_date_default and end_date_default and not selected_cells

    # Handle case when no data is available after filtering or filters are at default
    if df_filtered.empty or filters_at_default:
        selected_cell_text = "No data available for the selected date range."
        tweets_data = []

        # Empty plots
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
               base_distance_map_2d, base_distance_map_3d, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, selected_dates_store

    # -------------------- Adjustment Ends Here --------------------

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
    
    # Update tweets data for the table
    tweets_data_df = df_filtered.loc[selected_indices, ['text', 'engagement', 'engagement_log']]
    tweets_data_df.rename(columns={'text': 'tweet'}, inplace=True)
    tweets_data = tweets_data_df.to_dict('records')
    
    # Create sentiment scatter plot for selected tweets
    sentiment_plot = px.scatter(
        sentiment_selected,
        x='Sentiment Score',
        y='Engagement',
        color='Handle',
        title='Sentiment Polarity vs Total Engagement (Filtered Data)',
        labels={'Sentiment Score': 'Sentiment Polarity', 'Engagement': 'Total Engagement'}
    )
    sentiment_plot.update_layout(template="plotly_white")
    
    # Create line chart for selected data (by Handle)
    if not sentiment_selected.empty:
        # Ensure 'Date' column has valid dates
        if sentiment_selected['Date'].isna().any():
            sentiment_selected = sentiment_selected.dropna(subset=['Date'])
    
        sentiment_time_series = sentiment_selected.groupby(['Date', 'Handle'])['Sentiment Score'].mean().unstack()
        sentiment_time_series = sentiment_time_series.replace(0, np.nan)
    
        fig_line = go.Figure()
        for handle in sentiment_time_series.columns:
            fig_line.add_trace(
                go.Scatter(
                    x=sentiment_time_series.index,  # Dates
                    y=sentiment_time_series[handle],  # Average sentiment score for each handle
                    mode='lines+markers',  # Add markers to make single data points visible
                    name=handle
                )
            )
        fig_line.update_layout(
            title='Average Sentiment Analysis Over Time by Handle (Filtered Data)',
            xaxis_title='Date',
            yaxis_title='Average Sentiment Score',
            legend_title='Handles',
            hovermode='x',
            template='plotly_white'
        )
    
        # Create overall line chart for selected data
        sentiment_overall_time_series = sentiment_selected.groupby('Date')['Sentiment Score'].mean()
    
        fig_line_overall = go.Figure()
        fig_line_overall.add_trace(
            go.Scatter(
                x=sentiment_overall_time_series.index,
                y=sentiment_overall_time_series.values,
                mode='lines+markers',
                name='Overall Sentiment'
            )
        )
        fig_line_overall.update_layout(
            title='Average Sentiment Analysis Over Time (Filtered Data)',
            xaxis_title='Date',
            yaxis_title='Average Sentiment Score',
            hovermode='x',
            template='plotly_white'
        )
    else:
        # If no data, create empty figures
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
    
        fig_line_overall = go.Figure()
        fig_line_overall.add_annotation(
            text="No data available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=20)
        )
    
    # Update t-SNE plot
    mask_tsne = vis_df_tsne_filtered.index.isin(selected_indices)
    fig_tsne = px.scatter(
        vis_df_tsne_filtered,
        x='x',
        y='y',
        color='engagement_log',
        color_continuous_scale='Viridis',
        hover_data=['tweet', 'engagement'],
        title='t-SNE Visualization (Log Engagement)',
        width=600,
        height=600
    )
    if selected_cells:
        fig_tsne.update_traces(marker=dict(opacity=0.2))
        fig_tsne.add_trace(
            go.Scatter(
                x=vis_df_tsne_filtered.loc[mask_tsne, 'x'],
                y=vis_df_tsne_filtered.loc[mask_tsne, 'y'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle-open',
                    line=dict(width=2, color='DarkSlateGrey'),
                    opacity=1.0
                ),
                name='Selected Tweets',
                hoverinfo='text',
                text=vis_df_tsne_filtered.loc[mask_tsne, 'tweet']
            )
        )
    else:
        fig_tsne.update_traces(marker=dict(opacity=1.0))
    
    # Update UMAP plot
    mask_umap = vis_df_umap_filtered.index.isin(selected_indices)
    fig_umap = px.scatter(
        vis_df_umap_filtered,
        x='x',
        y='y',
        color='engagement_log',
        color_continuous_scale='Viridis',
        hover_data=['tweet', 'engagement'],
        title='UMAP Visualization (Log Engagement)',
        width=600,
        height=600
    )
    if selected_cells:
        fig_umap.update_traces(marker=dict(opacity=0.2))
        fig_umap.add_trace(
            go.Scatter(
                x=vis_df_umap_filtered.loc[mask_umap, 'x'],
                y=vis_df_umap_filtered.loc[mask_umap, 'y'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle-open',
                    line=dict(width=2, color='DarkSlateGrey'),
                    opacity=1.0
                ),
                name='Selected Tweets',
                hoverinfo='text',
                text=vis_df_umap_filtered.loc[mask_umap, 'tweet']
            )
        )
    else:
        fig_umap.update_traces(marker=dict(opacity=1.0))
    
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
            html.P("No cells selected."),
            html.P(f"Number of Tweets in Date Range: {len(df_filtered)}")
        ])
    
    # Update distance-map-2d with selected cells highlighted in black
    fig_distance_map_2d = base_distance_map_2d
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
    fig_distance_map_3d = base_distance_map_3d
    if selected_cells:
        selected_z = [distance_map[y, x] for x, y in selected_cells]
        selected_x_3d = [x for x, y in selected_cells]
        selected_y_3d = [y for x, y in selected_cells]
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
                text=[f"SOM X: {x}, SOM Y: {y}, Distance: {distance_map[y, x]:.2f}"
                      for (x, y), distance in zip(selected_cells, selected_z)]
            )
        )
    
    # ------------------- Generate Horizontal Word Frequency Bar Chart -------------------
    
    if not tweets_data_df.empty:
        # Extract all words from selected tweets
        all_words = ' '.join(tweets_data_df['tweet']).split()
    
        # Count word frequencies
        word_counts = Counter(all_words)
    
        # Get the top 20 words
        top_words = word_counts.most_common(20)
    
        if top_words:
            words, counts = zip(*top_words)
            # Create the horizontal bar chart
            fig_word_freq = px.bar(
                y=words,            # Assign words to the y-axis for horizontal bars
                x=counts,           # Assign counts to the x-axis
                labels={'x': 'Frequency', 'y': 'Words'},  # Update labels accordingly
                title='Top 20 Word Frequencies in Selected Tweets',
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
    
    # ------------------- Generate Engagement Box Plot -------------------
    
    if not tweets_data_df.empty:
        # Extract engagement data from selected tweets
        selected_engagement = tweets_data_df['engagement']
    
        if not selected_engagement.empty:
            # Create the box plot
            fig_box = px.box(
                y=selected_engagement,
                labels={'y': 'Engagement'},
                title='Engagement Distribution of Selected Tweets',
                points='outliers'  # Show outliers
            )
            fig_box.update_layout(
                plot_bgcolor='white',
                margin=dict(l=100, r=50, t=50, b=50)
            )
        else:
            # If no engagement data is present
            fig_box = go.Figure()
            fig_box.add_annotation(
                dict(
                    text="No engagement data to display.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=20)
                )
            )
            fig_box.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
    else:
        # If no tweets are selected, display an empty box plot with a message
        fig_box = go.Figure()
        fig_box.add_annotation(
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
        fig_box.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    
    # Return all plots and data along with the updated selected_dates_store
    return fig_tsne, fig_umap, selected_cell_text, tweets_data, \
           fig_distance_map_2d, fig_distance_map_3d, fig_word_freq, fig_box, sentiment_plot, fig_line, fig_line_overall, selected_dates_store

@app.callback(
    [
        Output('selected-cells-store', 'data'),
        Output('date-picker-range', 'start_date'),
        Output('date-picker-range', 'end_date')
    ],
    [
        Input('distance-map-2d', 'clickData'),
        Input('distance-map-3d', 'clickData'),
        Input('clear-selection-button', 'n_clicks'),
        Input('reset-date-filter-button', 'n_clicks')
    ],
    [
        State('selected-cells-store', 'data'),
        State('selected-dates-store', 'data')
    ]
)
def update_selected_cells_and_date(clickData_2d, clickData_3d, clear_clicks, reset_date_clicks, selected_cells, selected_dates_store):
    """
    Updates the list of selected cells and the date picker based on user interactions.
    Handles clearing selections and resetting the date filter.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default start and end dates
    start_date = pd.to_datetime(selected_dates_store['start_date']).date()
    end_date = pd.to_datetime(selected_dates_store['end_date']).date()

    if triggered_id == 'clear-selection-button':
        # Reset selected cells and date picker to full range
        selected_cells = []
        start_date = df['date'].min()
        end_date = df['date'].max()
        return selected_cells, start_date, end_date

    elif triggered_id == 'reset-date-filter-button':
        # Reset date picker to full range
        start_date = df['date'].min()
        end_date = df['date'].max()
        # Optionally, you can decide whether to reset selected_cells or keep them
        return selected_cells, start_date, end_date

    elif triggered_id == 'distance-map-2d' or triggered_id == 'distance-map-3d':
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

    else:
        # If no recognized input triggered the callback, prevent update
        raise PreventUpdate

# ------------------- Running the Dash App -------------------

if __name__ == '__main__':
    app.run_server(debug=True)
