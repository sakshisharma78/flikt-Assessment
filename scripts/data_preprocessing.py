import argparse
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()




def clean_text(text: str) -> str:
if not isinstance(text, str):
return ''
# remove special chars except punctuation
text = re.sub(r'[^\w\s\.!?,\'"-]', ' ', text)
text = text.replace('\n',' ').strip()
# lower
text = text.lower()
return text




def tokenize_lemmatize(text: str) -> str:
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
tokens = [t for t in tokens if t.isalpha()]
tokens = [t for t in tokens if t not in STOPWORDS]
tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
return ' '.join(tokens)




def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
df = df.drop_duplicates(subset=['text'])
df['text_clean'] = df['text'].fillna('').apply(clean_text)
df['text_tokens'] = df['text_clean'].apply(tokenize_lemmatize)
df = df[df['text_tokens'].str.strip() != '']
return df




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()


df = pd.read_csv(args.input)
df_clean = preprocess_df(df)
df_clean.to_csv(args.output, index=False)
print('Saved cleaned data to', args.output)