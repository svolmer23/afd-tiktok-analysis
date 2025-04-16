
# --- SETUP ---
# Install required libraries
# pip install pandas spacy scikit-learn matplotlib kneed wordcloud nltk seaborn emojis python-louvain

# Download necessary resources
import nltk
nltk.download('stopwords')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import networkx as nx
from collections import Counter
from itertools import combinations
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import emojis
import community as community_louvain

# --- DATA LOAD ---
df = pd.read_csv('Final_AfD_Dataset.csv', index_col=0)

# --- PREPROCESSING ---
df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
df['hashtag_list'] = df['hashtags'].apply(lambda x: [tag.strip().lower() for tag in str(x).split(',')] if pd.notnull(x) else [])
df['hashtags_clean'] = df['hashtag_list']

# --- WORDCLOUDS ---
def plot_wordcloud(text, stopwords, title=''):
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white', colormap='viridis', stopwords=stopwords
    ).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

for column in ['hashtags', 'stickers', 'body']:
    combined_text = ' '.join(df[column].astype(str))
    plot_wordcloud(combined_text, STOPWORDS.union({'nan', 'hashtag', 'sticker'}), f'WordCloud: {column}')

# --- TIMELINE OF RELEVANT POSTS ---
keywords = ['afd', '💙', 'weidel', 'alternative', 'alice', 'mutzurwahrheit', 'rechts', '🇩🇪']
pattern = '|'.join(re.escape(k) for k in keywords)
mask = df[['body', 'hashtags', 'author', 'stickers']].apply(lambda col: col.str.contains(pattern, case=False, na=False)).any(axis=1)
filtered_df = df[mask].dropna(subset=['date_posted'])

# Plot timeline
timeline = filtered_df.groupby(filtered_df['date_posted'].dt.date).size()
timeline.plot(kind='bar', figsize=(16, 6))
plt.title('Timeline of Relevant Posts')
plt.ylabel('Posts')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- AUTHOR FOLLOWER ANALYSIS ---
bins = [0, 1000, 10000, 100000, 1000000, 10000000, float('inf')]
labels = ['1-1k', '1k-10k', '10k-100k', '100k-1m', '1m-10m', '10m+']
df_unique_authors = filtered_df.drop_duplicates(subset='author')
df_unique_authors['follower_bins'] = pd.cut(df_unique_authors['author_followers'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
sns.countplot(data=df_unique_authors, x='follower_bins', palette='Set2', order=labels)
plt.xlabel('Follower Group')
plt.ylabel('Count of Authors')
plt.title('Follower Group Distribution')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\nNumber of authors in each follower group:")
print(df_unique_authors['follower_bins'].value_counts().sort_index())
print(f"Total unique authors considered: {df_unique_authors.shape[0]}")

# --- TOP AUTHORS BY POST COUNT ---
author_post_counts = filtered_df.groupby('author').size().reset_index(name='post_count')
author_post_counts_sorted = author_post_counts.sort_values(by='post_count', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(range(len(author_post_counts_sorted)), author_post_counts_sorted['post_count'], color='skyblue')
plt.xlabel('Authors')
plt.ylabel('Post Count')
plt.title('All Authors with Posts Containing Specified Keywords')
plt.xticks([])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Top 20 authors with the most posts:")
print(author_post_counts_sorted.head(20))

# --- PIE CHART OF TOP AUTHOR CATEGORIES ---
top_authors = author_post_counts_sorted.head(30)['author'].tolist()
author_categories = {
    'alice_weidel_afd': 'official account',
    'real.germ4n': 'fan account',
    'breaking_2025': 'fan account',
    'maxey_2': 'anti fan account',
    'alice_weidel_fan': 'fan account',
    'real_german 1': 'fan account',
    'afdfraktionimbundestag': 'official account',
    'politikdog': 'unofficial news outlet',
    'rtlaktuell': 'news outlet',
    'revo.2025': 'fan account',
    'mrs.burnout2.0': 'fan account',
    'politikundheimat': 'fan account',
    'deutscherpatriot.1': 'fan account',
    'andreazuercherafd': 'official account',
    'cduenjoyer': 'anti fan account',
    'anna.nguyen.afd': 'official account',
    'politik.edit': 'anti afd account',
    'kawaiiwg': 'anti fan account',
    'spillthetea2025': 'anti fan account',
    'issyb0': 'fan account',
    'sirschweiger089': 'fan account',
    'politikerclips': 'user not found',
    'doctorzenedine': 'fan account',
    'afd_edit5': 'fan account',
    'kate59385': 'not AfD related',
    'afd.nachrichten': 'fan account',
    'johannaruediger': 'unofficial news outlet',
    'heimatecho': 'fan account',
    'fettbaer88': 'fan account'
}

df_pie = pd.DataFrame({
    'Author': top_authors,
    'Category': [author_categories.get(author, 'unknown') for author in top_authors]
})

category_counts = df_pie['Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Account Categories among Top 30 Authors")
plt.axis('equal')
plt.tight_layout()
plt.show()

# --- TOP HASHTAGS ---
all_hashtags_list = df['hashtags'].dropna().astype(str).str.split(',').explode().str.strip()
hashtag_counts = Counter(all_hashtags_list)
print("Top 50 most used hashtags:")
for tag, count in hashtag_counts.most_common(50):
    print(f"{tag}: {count}")

# --- AFD VS ANTI-AFD TIMELINE ---
afd_tags = ['afd', '💙', 'aliceweidel', 'weidel', 'alternativefürdeutschland', 'afddeutschland', 'teamalice', 'deshalbafd', 'jetztafd', 'afdfraktion', 'alice', 'mutzurwarheit']
anti_afd_tags = ['noafd', 'fckafd', 'gegenrechts']

df_afd = df[df['hashtags_clean'].apply(lambda tags: any(tag in tags for tag in afd_tags) and not any(tag in tags for tag in anti_afd_tags))]
df_anti_afd = df[df['hashtags_clean'].apply(lambda tags: any(tag in tags for tag in anti_afd_tags))]

timeline_afd = df_afd.groupby(df_afd['date_posted'].dt.date).size().reset_index(name='AfD-related posts')
timeline_anti = df_anti_afd.groupby(df_anti_afd['date_posted'].dt.date).size().reset_index(name='Anti-AfD posts')
timeline_combined = pd.merge(timeline_afd, timeline_anti, on='date_posted', how='outer').fillna(0)
timeline_combined['date_posted'] = pd.to_datetime(timeline_combined['date_posted'])

plt.figure(figsize=(12, 6))
plt.plot(timeline_combined['date_posted'], timeline_combined['AfD-related posts'], label='AfD-related posts', marker='o')
plt.plot(timeline_combined['date_posted'], timeline_combined['Anti-AfD posts'], label='Anti-AfD posts', marker='s')
plt.title("AfD vs Anti-AfD Post Timeline")
plt.xlabel("Date")
plt.ylabel("Number of Posts")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- RECURRING MUSIC IDS ---
music_counts = df['music_id'].value_counts()
recurring_music_df = music_counts[music_counts >= 4].reset_index()
recurring_music_df.columns = ['music_id', 'count']
print("\nRecurring music IDs with counts >= 4:")
print(recurring_music_df.sort_values(by='count', ascending=False))

