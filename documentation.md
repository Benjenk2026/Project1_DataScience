# Data Cleaning Pipeline

This file explains how to use cleaning.py to clean the datasets for the project.

## Overview

The cleaning.py script is used to clean all raw data files and output cleaned versions into the processed folder.

It uses pandas and basic data cleaning techniques from class such as
standardizing column names
handling missing values
removing duplicates
simple text normalization
basic location cleaning

The goal is to make all datasets consistent and ready for the next phase.

## File Locations

Input files should be placed in  
data/raw/

Cleaned output files will be saved to  
data/processed/

## How to Run

Run from the root of the project.

### Clean one file
python3 src/cleaning.py --file 311

Example options for file
311
yelp_business
yelp_review
yelp_user
yelp_tip
yelp_checkin

## Chunked Mode

Chunked mode is used for very large files that cannot fit into memory.

Instead of loading the entire dataset at once, the file is processed in smaller batches.

Each batch is cleaned and written to the output file before moving to the next batch.

## When to use it

Use chunked mode for large files such as
yelp_review
yelp_user

## How to run chunked mode
python3 src/cleaning.py --file yelp_review --chunked

If running all files, chunked mode is automatically used for large datasets.

## What the Cleaning Does

For each dataset, the script performs

Converts column names to snake_case
Trims whitespace and standardizes text
Drops rows missing important fields
Removes columns with too many missing values
Fills missing values where appropriate
Removes duplicate records based on ID
Cleans latitude and longitude values
Applies simple rule based grouping for categories


## Sample and Preview Data

A sample version of the cleaned data is included to keep file sizes small and easy to work with.

The sample data is limited to stay under size constraints and represents a portion of the full dataset.

A preview version is also included which contains only a few hundred rows.

This is useful for quickly checking structure, columns, and results without running the full pipeline.


## Output

Each cleaned file is saved as a CSV in data/processed/ with a consistent naming format.

Example
311_cleaned.csv
yelp_business_cleaned.csv

These cleaned files are used in the next phase of the project.

## Notes

The script is designed to be simple and follow concepts covered in class
Chunked mode trades some accuracy for performance but allows large files to be processed
Make sure required libraries are installed before running
