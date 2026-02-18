import sys
sys.path.insert(0,'src')
import cleaning

df = cleaning.openfile(r'data/raw/yelp_JSON_test.json')
print('COLUMNS:', df.columns.tolist())
print('SHAPE:', df.shape)
from pprint import pprint
pprint(df.head(5).to_dict(orient='records'))
