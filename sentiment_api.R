# sentiment_api.R — Sentiment Analysis REST API
# Run: Rscript sentiment_api.R
#
# Endpoints:
#   GET  /health   — health check
#   GET  /sample   — predict on built-in sample reviews
#   POST /predict  — predict on custom texts
#                    Body: {"texts": ["review 1", "review 2", ...]}

# ── Dependencies ───────────────────────────────────────────────────────────────
required <- c("plumber", "glmnet", "tm", "Matrix", "jsonlite")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}

library(plumber)
library(glmnet)
library(tm)
library(Matrix)
library(jsonlite)

# ── Load model ─────────────────────────────────────────────────────────────────
model_path <- "sentiment_model.rds"
if (!file.exists(model_path)) stop("sentiment_model.rds not found.")

model <- readRDS(model_path)
cat(sprintf("Model loaded  | lambda.min = %.6f\n", model$lambda.min))

# Extract vocabulary from model coefficient matrix
vocab <- rownames(coef(model, s = "lambda.min")[[1]])
vocab <- vocab[vocab != "(Intercept)"]
cat(sprintf("Vocabulary    | %d terms\n\n", length(vocab)))

# ── Text preprocessing (mirrors training pipeline) ─────────────────────────────
preprocess <- function(texts) {
  corpus <- VCorpus(VectorSource(as.character(texts)))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  corpus <- tm_map(corpus, stripWhitespace)
  dtm    <- DocumentTermMatrix(corpus, control = list(dictionary = vocab))
  Matrix::sparseMatrix(
    i = dtm$i, j = dtm$j, x = as.numeric(dtm$v),
    dims = dim(dtm), dimnames = dimnames(dtm)
  )
}

# ── Prediction helper ──────────────────────────────────────────────────────────
predict_sentiment <- function(texts) {
  x          <- preprocess(texts)
  pred_class <- as.character(predict(model, newx = x, s = "lambda.min", type = "class"))
  pred_prob  <- predict(model, newx = x, s = "lambda.min", type = "response")

  lapply(seq_along(texts), function(i) {
    list(
      id        = i,
      text      = texts[[i]],
      sentiment = pred_class[i],
      probabilities = list(
        negative = round(pred_prob[i, "negative", 1], 4),
        neutral  = round(pred_prob[i, "neutral",  1], 4),
        positive = round(pred_prob[i, "positive", 1], 4)
      )
    )
  })
}

# ── Sample dataset ─────────────────────────────────────────────────────────────
sample_reviews <- c(
  "This product is absolutely amazing! Best purchase I have ever made.",
  "Terrible quality. Broke after one day. Complete waste of money.",
  "It is okay, nothing special. Does what it is supposed to do.",
  "I love this item. Fast shipping and great packaging.",
  "Very disappointed. The color was wrong and the size was too small.",
  "Average product at an average price. Not bad, but not great either.",
  "Highly recommend this to everyone. Works perfectly out of the box.",
  "I returned it immediately. The description was completely misleading."
)

# ── Build API ──────────────────────────────────────────────────────────────────
pr <- plumber::pr()

# Health check
pr$handle("GET", "/health", function() {
  list(
    status     = "ok",
    model      = "glmnet-multinomial-sentiment",
    classes    = c("negative", "neutral", "positive"),
    lambda_min = round(model$lambda.min, 6),
    vocab_size = length(vocab)
  )
})

# Sample dataset prediction
pr$handle("GET", "/sample", function() {
  predictions <- predict_sentiment(sample_reviews)
  list(
    count       = length(sample_reviews),
    predictions = predictions
  )
})

# Custom text prediction
pr$handle("POST", "/predict", function(req, res) {
  tryCatch({
    body  <- jsonlite::fromJSON(req$postBody, simplifyVector = FALSE)
    texts <- body$texts

    if (is.null(texts) || length(texts) == 0) {
      res$status <- 400
      return(list(error = "Request body must contain a non-empty 'texts' array."))
    }

    predictions <- predict_sentiment(unlist(texts))
    list(count = length(texts), predictions = predictions)

  }, error = function(e) {
    res$status <- 500
    list(error = conditionMessage(e))
  })
})

# ── Start server ───────────────────────────────────────────────────────────────
port <- as.integer(Sys.getenv("PORT", "8000"))

cat(sprintf("Sentiment API starting on http://0.0.0.0:%d\n", port))
cat("  GET  /health   — health check\n")
cat("  GET  /sample   — built-in sample reviews\n")
cat("  POST /predict  — custom texts\n")
cat(sprintf('\nExample: curl -X POST http://localhost:%d/predict \\\n', port))
cat('           -H "Content-Type: application/json" \\\n')
cat('           -d \'{"texts":["Amazing product!","Terrible, avoid."]}\'\n\n')

pr$run(host = "0.0.0.0", port = port, docs = FALSE)
