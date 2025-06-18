# Load required packages
pkgs <- c("rvest", "tm", "textclean", "textstem", "topicmodels", 
          "tidytext", "ggplot2", "dplyr", "reshape2", 
          "pheatmap", "wordcloud", "RColorBrewer")

# Install any missing packages
if (length(setdiff(pkgs, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(pkgs, rownames(installed.packages())))
}

# Load all required libraries
lapply(pkgs, library, character.only = TRUE)

# List of BBC base URLs to scrape
base_urls <- c(
  "https://www.bbc.com/news",
  "https://www.bbc.com/sport",
  "https://www.bbc.com/business",
  "https://www.bbc.com/innovation",
  "https://www.bbc.com/culture",
  "https://www.bbc.com/arts",
  "https://www.bbc.com/travel",
  "https://www.bbc.com/future-planet",
  "https://www.bbc.com/video",
  "https://www.bbc.com/sport/football"
)

# Function to discover article links from a BBC page
discover_links <- function(url) {
  page <- read_html(url)
  selectors <- c(".gs-c-promo-heading", "a.qa-heading-link", "a[href*='/news/']")
  links <- unique(unlist(lapply(selectors, function(sel) {
    page %>% html_elements(css = sel) %>% html_attr("href")
  })))
  links <- links[!is.na(links)]
  # Filter only specific sections
  links <- links[grepl("^(/news|/sport|/business|/culture|/travel|/future-planet|/arts|/video|/innovation|/sport/football)", links)]
  # Convert relative URLs to absolute URLs
  links <- ifelse(grepl("^https?://", links), links, paste0("https://www.bbc.com", links))
  return(links)
}

# Scrape and consolidate all links from the given BBC sections
all_links <- unique(unlist(lapply(base_urls, discover_links)))
news_links <- head(all_links, 50)  # Limit to first 50 links for analysis

# Function to preprocess text data
preprocess_text <- function(text) {
  text <- tolower(text)                     # Convert to lowercase
  text <- replace_contraction(text)         # Replace contractions (e.g., don't → do not)
  text <- replace_emoji(text)               # Remove emojis
  text <- gsub("<[^>]+>", "", text)         # Remove HTML tags
  text <- gsub("[^a-z ]", " ", text)        # Keep only alphabetic characters
  text <- gsub("\\s+", " ", text)           # Remove extra spaces
  return(trimws(text))                      # Trim leading/trailing whitespace
}

# Initialize vectors to store data
original_articles <- c()
processed_articles <- c()
successful_links <- c()

# Loop through article links and extract content
for (link in news_links) {
  try({
    page <- read_html(link)
    # Try multiple selectors to extract paragraphs
    paras <- html_elements(page, "div[data-component='article-body'] p, .article-body-commercial-selector p")
    if (length(paras) == 0) paras <- html_elements(page, "p")  # Fallback to generic paragraphs
    full_text <- paste(html_text(paras), collapse = " ")
    cleaned <- preprocess_text(full_text)
    if (nzchar(cleaned)) {
      original_articles <- c(original_articles, full_text)
      processed_articles <- c(processed_articles, cleaned)
      successful_links <- c(successful_links, link)
    }
    Sys.sleep(1)  # Politeness delay to avoid hammering the server
  }, silent = TRUE)
}

# Convert the cleaned articles into a corpus for text mining
corpus <- VCorpus(VectorSource(processed_articles))
corpus <- tm_map(corpus, removeWords, stopwords("en"))             # Remove stopwords
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))   # Apply lemmatization
corpus <- tm_map(corpus, stripWhitespace)                          # Remove extra whitespace
final_texts <- sapply(corpus, as.character)

# Create a dataframe of article data
articles_df <- data.frame(
  ID = seq_along(final_texts),
  Article_URL = successful_links,
  Article_Content = final_texts,
  stringsAsFactors = FALSE
)

# Save the processed articles to a CSV file
write.csv(articles_df, "Processed_Articles.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("CSV file 'Processed_Articles.csv' created.\n")

# Create Document-Term Matrix
dtm <- DocumentTermMatrix(VCorpus(VectorSource(final_texts)))
row_totals <- apply(dtm, 1, sum)
dtm <- dtm[row_totals > 0, ]  # Remove empty documents

# Create a TF-IDF weighted matrix (optional - not used in LDA)
dtm_tfidf <- weightTfIdf(dtm)

# Apply LDA topic modeling (k = 5 topics)
k <- 5
lda_model <- LDA(dtm, k = k, control = list(seed = 1234))

# Get top terms (words) in each topic
lda_tidy <- tidy(lda_model, matrix = "beta")
top_terms <- lda_tidy %>%
  group_by(topic) %>%
  slice_max(order_by = beta, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(topic, -beta)

print(top_terms)

# Get dominant topic for each document
doc_topic <- tidy(lda_model, matrix = "gamma")
dominant_topic <- doc_topic %>%
  group_by(document) %>%
  slice_max(order_by = gamma, n = 1)

print(dominant_topic)

# Merge topic assignment back to original data
articles_df$Dominant_Topic <- dominant_topic$topic[match(articles_df$ID, as.numeric(dominant_topic$document))]
articles_df$Topic_Probability <- dominant_topic$gamma[match(articles_df$ID, as.numeric(dominant_topic$document))]

# Extract top keywords per topic
topic_keywords <- lda_tidy %>%
  group_by(topic) %>%
  slice_max(beta, n = 3) %>%
  summarise(keywords = paste(term, collapse = ", "))

# Add keywords and word count to dataset
articles_df$Top_Keywords <- topic_keywords$keywords[match(articles_df$Dominant_Topic, topic_keywords$topic)]
articles_df$Word_Count <- sapply(strsplit(articles_df$Article_Content, "\\s+"), length)

# Assign descriptive topic labels (manually mapped)
topic_labels <- c(
  "Economy & Finance",
  "Sports",
  "Politics",
  "Technology",
  "Health",
  "Entertainment",
  "Travel",
  "Environment",
  "Culture",
  "Science"
)
articles_df$Category <- topic_labels[articles_df$Dominant_Topic]

# Save final enriched dataset
write.csv(articles_df, "Processed_Articles_with_Categories.csv", row.names = FALSE)
cat("Final CSV 'Processed_Articles_with_Categories.csv' saved.\n")

# Visualize article count per category
ggplot(articles_df, aes(x = Category, fill = Category)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Article Count by Category", x = "Category", y = "Number of Articles") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Visualize top terms in each topic
ggplot(top_terms, aes(x = reorder(term, beta), y = beta, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Terms in Each Topic", x = "Terms", y = "Beta") +
  theme_minimal()

# Visualize number of documents per topic
dominant_topic %>%
  count(topic) %>%
  ggplot(aes(x = as.factor(topic), y = n, fill = as.factor(topic))) +
  geom_bar(stat = "identity") +
  labs(title = "Document Count per Dominant Topic", x = "Topic", y = "Count") +
  theme_minimal()

# Generate word cloud from TF-IDF scores
tfidf_matrix <- as.matrix(dtm_tfidf)
word_freq <- colSums(tfidf_matrix)
word_freq <- sort(word_freq, decreasing = TRUE)
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)

# Wordcloud visualization
set.seed(1234)
wordcloud(words = word_freq_df$word, freq = word_freq_df$freq,
          min.freq = 3, scale = c(3, 0.5), colors = brewer.pal(8, "Dark2"))

