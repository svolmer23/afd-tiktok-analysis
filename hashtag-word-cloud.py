# Combine all hashtags into a single string
all_hashtags = ' '.join(df['hashtags'].astype(str))

# Define custom stopwords to exclude certain words
custom_stopwords = {'nan', 'nan nan'}

# Create the WordCloud object with the specified parameters and the custom stopwords
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',
    stopwords=custom_stopwords
).generate(all_hashtags)

# Display the WordCloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
