# --- SETUP ---
# pip install pandas matplotlib nltk wordcloud spacy scikit-learn kneed

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- LOAD DATA ---
df = pd.read_csv('UnavailablePosts.csv', index_col=0)

# --- WORDCLOUD FOR STICKERS ---
all_stickers = ' '.join(df['stickers'].astype(str))
german_stopwords = set(stopwords.words('german'))
custom_stopwords = {'hashtag', 'sticker', 'nan', 'nan nan'}
all_stopwords = STOPWORDS.union(german_stopwords).union(custom_stopwords)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=all_stopwords,
    colormap='viridis'
).generate(all_stickers)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.title("WordCloud of Stickers in Unavailable Posts")
plt.show()

# --- VIEWS BAR CHART (17–23 Feb 2025) ---
df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
mask = (df['date_posted'] >= '2025-02-17') & (df['date_posted'] <= '2025-02-23')
filtered_df = df[mask].sort_values(by='plays').reset_index(drop=True)

plt.figure(figsize=(14, 6))
plt.bar(filtered_df.index, filtered_df['plays'], color='skyblue', label='Views per post')
plt.axhline(y=660555.04, color='red', linestyle='--', linewidth=2, label='Avg plays (AfD-related posts)')
plt.title("Views of Unavailable Posts (17–23 Feb 2025)")
plt.xlabel("Post Index")
plt.ylabel("Number of Plays")
plt.legend()
plt.tight_layout()
plt.show()

# --- POST COUNTS PER DAY ---
df = df.dropna(subset=['date_posted'])
df['upload_date'] = df['date_posted'].dt.date
date_counts = df['upload_date'].value_counts().sort_index()

plt.figure(figsize=(14, 6))
plt.bar(date_counts.index.astype(str), date_counts.values, color='cornflowerblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Upload Date")
plt.ylabel("Number of Unavailable Posts")
plt.title("Number of Unavailable Posts by Upload Date")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

