# City of Philadelphia 
| Dataset File      | Description                       | Size                  | Record Count       | Format        |
| ----------------- | --------------------------------- | --------------------- | ------------------ | ------------- |
| philidelphia.xlsx | Philadelphia 311 service requests | Small (< 1 MB sample) | 8 records (sample) | Excel (.xlsx) |

### Data format
The dataset is structured in a relational tabular format, unlike the Yelp dataset’s semi-structured JSON.


# Yelp Datasets
| Dataset File   | Description          | Approx Size | Record Count (Typical) | Format                |
| -------------- | -------------------- | ----------- | ---------------------- | --------------------- |
| business.json  | Business information | ~200 MB     | ~150k records          | JSON (line-delimited) |
| review.json    | Customer reviews     | ~5–7 GB     | ~6–7 million records   | JSON                  |
| user.json      | User profiles        | ~2–3 GB     | ~2 million records     | JSON                  |
| checkin.json   | Check-in timestamps  | ~100 MB     | ~150k records          | JSON                  |
| tip.json       | Short user tips      | ~200 MB     | ~1 million records     | JSON                  |


### Data Formats
Primary Format: JSON Lines

All datasets use a line-delimited JSON format, where:
- Each row is a standalone JSON object
- Nested objects and arrays are common
- Some fields contain complex structures (dictionaries or lists)
- This semi-structured format enables flexibility but requires preprocessing for relational or tabular analysis.

### Schema Differneces 

### Structural differences
| Yelp Dataset              | Philadelphia Dataset   |
| ------------------------- | ---------------------- |
| Semi-structured JSON      | Fully structured table |
| Nested objects and arrays | Flat schema            |
| Multiple linked files     | Single table           |

### Data Content Differences
| Yelp                      | Philadelphia                   |
| ------------------------- | ------------------------------ |
| Reviews and business info | Government service operations  |
| Social network data       | Public complaint workflow      |
| Text-heavy sentiment data | Administrative service records |

### Relationship Differences
- Yelp uses relational links between multiple datasets
- Philadelphia dataset is mostly self-contained

### Representation Differences
Key inconsistencies:
- Many missing values in address and location fields
- Date fields stored as timezone-aware timestamps
- Some service notices are textual instead of numeric

### Data Integration Challenges
Major challenges when integrating with other datasets:
- Handling missing geographic information
- Standardizing timestamps
- Mapping service codes to categories
- Normalizing inconsistent text fields


