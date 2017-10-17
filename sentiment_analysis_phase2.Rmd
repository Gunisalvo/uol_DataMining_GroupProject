---
title: "Data Mining Group project: Ali, Li, Rodrigo"
output: html_notebook
---

### A. DEPENDENCIES:
library(devtools)
install_github("jeroen/mongolite")
install.packages('rvest')
install.packages('xml2')
install.packages('tidytext')
install.packages('tidyverse')
install_github("ggrothendieck/sqldf")
library(mongolite)
library(tidyverse)      # data manipulation & plotting
library(stringr)        # text cleaning and regular expressions
library(tidytext)       # provides additional text mining functions
library(magrittr)
library(sqldf)
library(gmodels)
library(class)
install.packages("KernelKnn")
library(KernelKnn)

### 1. LOAD:
# Connect to the DB, please replace username and password by the ones shared on the forum
con <- mongo("reviews", url = "mongodb://datamining:datamining@ds113445.mlab.com:13445/heroku_8tv2vqr4")

# Random generator to control sampling, 1000 samples
set.seed(123)
db_sample <- sample(37000, 1000)

# Load review sample
all_reviews <- con$find()

# Select sample
reviews <- all_reviews[db_sample,]

### 2. CLEAN UP:
# Start empty list of indexed words (chapter = review)
word_count <- tibble()

# Remove useless terms and punctuation
for(i in 1:nrow(reviews)){
  r <- reviews[i,]
  clean <- tibble(chapter=i, text=r$reviewText) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words)
    
  clean$score <- r$overall

  word_count <- rbind(word_count, clean)
}

# Group word per review (chapter = review)
word_count_breakdown <- word_count %>% group_by(chapter, word, score) %>% count(word)
colnames(word_count_breakdown)[colnames(word_count_breakdown) == 'n'] <- 'word_count'

# Join sentiment scores
score_breakdown <- word_count_breakdown %>%
  right_join(get_sentiments("nrc")) %>%
  filter(!is.na(sentiment)) %>%
  count(sentiment, sort = TRUE)
colnames(score_breakdown)[colnames(score_breakdown) == 'n'] <- 'word_count'

# Count sentiments by review
score_breakdown_clean <- sqldf("select chapter, sentiment, score, sum(word_count) as 'count' from score_breakdown where chapter <> 'NA' group by chapter, sentiment, score")

# The number of occuring sentiments is dynamic need this to initialize the variable length Matrix
sentiment_list <- (sqldf("select distinct(sentiment) from score_breakdown_clean where chapter <> 'NA' group by sentiment"))$sentiment

# Add missing columns
relevant_attributes <- c(c("review", "score"), sentiment_list)

# Build the sentiment matrix
scores <- as.data.frame(
  setNames(replicate(length(relevant_attributes), integer()),relevant_attributes[1:length(relevant_attributes)])
)
names(scores) <- relevant_attributes

# Initialize all the position
for(i in sqldf("select distinct(chapter) as 'chapter' from score_breakdown_clean")$chapter){
  s <- list()
  s$review <- i
  s$score <- sqldf(sprintf("select distinct(score) as 'score' from score_breakdown_clean where chapter = '%s' ", i))$score
  
  for(sent in sentiment_list){
    value <- sqldf(sprintf("select distinct(count) as 'count' from score_breakdown_clean where chapter = '%s' and sentiment = '%s' ", i, sent))$count
    if(length(value) == 0){
      s[[sent]] <- 0
    }else{
      s[[sent]] <- value
    }
  }
  scores <- rbind(scores, data.frame(s))
}

summary(scores[,-1])

### 3. ANALYSIS:

# Select Training (90%) and Test (10%) data 
train_sample <- sample(nrow(scores), nrow(scores) - (nrow(scores) %/% 10))

scores.train_scores <- scores[train_sample,]$score
scores.train <- scores[train_sample,][-1:-2]
scores.train_pn <- scores[train_sample,][9] - scores[train_sample,][8]
scores.test_scores <- scores[-train_sample,]$score
scores.test <- scores[-train_sample,][-1:-2]
scores.test_pn <- scores[-train_sample,][9] - scores[-train_sample,][9]

## 3.1 LINEAR REGRESSION (Positive - Negative = Sentiment):

# Plot the distibuition of Review Scores x  Sentiment (Positive - Negative words)
scatter.smooth(x=scores.train_scores, y=scores.train_pn$positive, xlab = "score", ylab = "sentiment" ,main="Sentiment ~ Score")


## 3.2 K Nearest Neighbours (Euclidean distance)

# Train the Classifier
scores.eucledian <- knn(train=scores.train, test=scores.test, cl=scores.train_scores, k=5)

# Confusion Matrix of the Classifier
CrossTable(x=scores.test_scores, y=scores.eucledian)
# 47/92 = 51.08%

## 3.3 K Nearest Neighbours (Cosine distance)

# Train the Classifier
scores.cosine <- KernelKnn(scores.train, TEST_data = scores.test, y = scores.train_scores, k = 10, method = 'canberra', weights_function = 'cosine', Levels = unique(scores.train_scores), regression=F)

# The results are shown as the probablility of belonging to a given class
head(scores.cosine)

# Select the Highest Probability
scores.cosine_result <- list()

for(i in 1:nrow(scores.cosine)){
  x <- NA
  if(!is.nan(max(scores.cosine[i,]))){
    x <- which(scores.cosine[i,]==max(scores.cosine[i,]))
  }
  scores.cosine_result <- rbind(scores.cosine_result, x)
}

# Confusion Matrix of the Classifier
CrossTable(x=scores.test_scores, y=unlist(scores.cosine_result))
## 34/68 = 50%