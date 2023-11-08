from sklearn.feature_extraction.text import TfidfVectorizer

#list of text documents
text = ['The quick brown fox jumped over the lazy dog',
        'The dog.',
        'The fox.']

vectorizer = TfidfVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)
print(vectorizer.idf_)

vector = vectorizer.transform([text[0]])
print(vector.shape)
print(vector.toarray())
