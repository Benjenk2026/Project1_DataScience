Normalize function changes 
    - Rules applied:
    - Lowercase all text
    - Replace spaces with _
    - Replace non-alphanumeric characters with _
    - Collapse multiple _ into one

    Common short tokens are expanded into full, consistent names.
    Examples:
    Abbreviation	Standardized
    lat  -->	latitude
    lon / long	--> longitude
    zip / zipcode / postcode  -->	postal_code
    addr -->	address
    id variations -->	id_* prefix standardized

    If multiple original columns normalize to the same name, they are merged:
    Combined into one column
    Uses the first non-null value (left → right)

    Semantic Standardization Rules

    Additional normalization based on column meaning.
    Geographic Fields
    Ensures consistent naming:
    * latitude
    * longitude

    Count Fields
    Columns representing counts are renamed with a num_ prefix:

    Boolean Detection
    Columns containing boolean-like values (0/1/true/false) are standardized with an is_ prefix when appropriate.

    Final Cleanup & Uniqueness
    A final pass ensures consistency:
    Removes remaining special characters
    Collapses duplicate underscores
    Enforces lowercase formatting
    Guarantees unique column names by appending numeric suffixes if needed



# MongoDB Text Analytics Project

## Overview
This project explores text analytics capabilities in **MongoDB** using a dataset of approximately **50,000 movie reviews**. The primary goal is to evaluate how well MongoDB supports common text analysis tasks such as keyword search, anomaly detection, frequency analysis, and document similarity using aggregation pipelines.

The focus of this project is **running, measuring, and interpreting queries**, rather than building an application.

---
Link to sample data https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
## Dataset
- **Collection name:** `reviews`
- **Document structure:**
```js
{
  _id: ObjectId(...),
  review: "<full text review>",
  sentiment: "positive" | "negative"
}
```
- **Document count:** ~50,000
- Data imported from a CSV file using **MongoDB Compass**.

---
## Environment & Tools
- MongoDB local (Atlas should work the same, I just ran local)
- MongoDB Compass (used for running and visualizing aggregation pipelines)
- Text index created on the `review` field

```js
db.reviews.createIndex({ review: "text" })
```

---

## Queries Executed

### 1. Keyword / Phrase Search
**Goal:** Find all documents containing a specific keyword or phrase (e.g., "excellent").

**Query styles used:**
- Regex-based search (`$regex`)
- Full-text search using MongoDB text index (`$text`)

**Key observations:**
- `$text` search returns more results due to tokenization and normalization.
- Regex search is more literal but less scalable.

Execution time and result size were measured using:
```js
db.reviews
  .find({ review: { $regex: "\\bgood\\b", $options: "i" } })
```

---

### 2. Text Length Anomaly Detection
**Goal:** Identify unusually short or long reviews.

**Approach:**
- Computes the number of characters in each review using $strLenCP
- Filters the collection to return only documents with unusually short reviews (less than 50 characters) or unusually long reviews (greater than 1000 characters)

Regex search (simple & direct)
```js
db.reviews.aggregate([
  {
    $addFields: {
      textLength: { $strLenCP: "$review" }
    }
  },
  {
    $match: {
      $or: [
        { textLength: { $lt: 50 } },
        { textLength: { $gt: 1000 } }
    ]
  }
}
])
```
Text index search (best practice) requires Text index ran before hand 
```js
db.reviews.aggregate([
  {
    $project: {
      review: 1,
      textLength: { $strLenCP: "$review" }
  }
},
  {
    $bucketAuto: {
      groupBy: "$textLength",
      buckets: 10
  }
}
])
```

---

### 3. Most Frequent Words (Token Frequency)
**Goal:** Identify the most common words across all reviews.

**Approach:**
- Extract tokens using regex-based word matching
- Normalize to lowercase
- Aggregate word counts across the collection

  Full aggregation pipeline (classic NLP-style)
```js
db.reviews.aggregate([
  { $project: { words: { $split: [{ $toLower: "$review" }, " "] } } },
  { $unwind: "$words" },
  { $group: { _id: "$words", count: { $sum: 1 } } },
  { $sort: { count: -1 } },
  { $limit: 10 } ])
```
  Ignore very short tokens (cleaner results)
```js
db.reviews.aggregate([  { $project: { words: {
        $filter: {
          input: { $split: [{ $toLower: "$review" }, " "] },
          as: "w",
          cond: { $gte: [{ $strLenCP: "$$w" }, 4] } } } } },
  { $unwind: "$words" },
  { $group: { _id: "$words", count: { $sum: 1 } } },
  { $sort: { count: -1 } },
  { $limit: 10 }
])
```
Text index token frequency (limited but fast)
```js
db.reviews
  .find({ $text: { $search: "excellent good bad great terrible" } })
```

---

### 4. Document Similarity (Shared Keywords)
**Goal:** Identify pairs of documents that share **two or more common keywords**.

**Approach:**
1. Tokenize reviews into meaningful words
2. Build an inverted index (word → document IDs)
3. Generate document pairs per shared word
4. Count shared keywords per document pair

**Key implementation details:**
- This must be ran in Compass, within the 'Aggregation' tab
- Regex-based token extraction to avoid punctuation tokens
- Deduplication of words per document
- Use of `$slice` to cap documents per word
- `allowDiskUse: true` to prevent memory issues
- Limited to only 100 documents due to complexity

**Why this matters:**
- Demonstrates both the power and limitations of MongoDB for similarity analysis
- Highlights the risk of combinatorial explosion in document-pair queries
```js
[{ "$limit": 100 }, {
    "$addFields": {
      "keywords": { "$split": [ { "$toLower": "$review" }, " " ] } } }, {
    "$lookup": {
      "from": "reviews",
      "let": { "keywordsA": "$keywords", "idA": "$_id" },
      "pipeline": [
        { "$match": { "$expr": { "$lt": [ "$_id", "$$idA" ] } } },
        { "$addFields": { "keywords": { "$split": [ { "$toLower": "$review" }, " " ] } } },
        { "$addFields": { "commonKeywords": { "$setIntersection": [ "$keywords", "$$keywordsA" ] } }},{ "$match": { "commonKeywords.1": { "$exists": true } } },  { "$project": { "_id": 1, "review": 1, "commonKeywords": 1 } } ],"as": "matchingDocs"}
  },
  { "$unwind": "$matchingDocs" },
  {"$project": {"doc1_id": "$_id",
      "doc1_review": "$review",
      "doc2_id": "$matchingDocs._id",
      "doc2_review": "$matchingDocs.review",
      "commonKeywords": "$matchingDocs.commonKeywords"} }
]
```


---

## Performance Measurement
For each major query, the following were recorded:
- **Execution time** (`executionTimeMillis`)
- **Result size** (`nReturned` or summed bucket counts)

Measured using: add this to end of each query and it will return metrics instead of results
```js
.explain("executionStats")
```

---

## Key Findings
- MongoDB performs well for keyword search and aggregation-based analytics.
- Text indexing significantly improves search performance.
- Advanced text analysis requires careful preprocessing.
- Similarity detection is possible but not ideal at scale without external NLP tools.

---

## Limitations
- No built-in NLP features such as stemming or lemmatization
- Aggregation pipelines can become memory-intensive
- Tokenization is sensitive to punctuation and data formatting
- Some operators are version-dependent

---

## Future Work
- Integrate external NLP libraries for semantic similarity
- Apply stop-word removal and stemming
- Use approximate similarity methods (e.g., MinHash)
- Explore vector databases for large-scale text similarity

---

## Conclusion
This project demonstrates that MongoDB is a strong platform for **basic to intermediate text analytics**, particularly when queries are carefully designed and performance considerations are addressed. While not a replacement for full NLP pipelines, MongoDB provides valuable in-database text processing capabilities suitable for exploratory analysis and coursework.
