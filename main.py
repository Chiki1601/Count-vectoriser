from sklearn.feature_extraction.text import CountVectorizer

document = [&quot;My name is Pooja&quot;,
&quot;my name is Aditya&quot;,
&quot;Each Friend helps many other friends at

anywhere&quot;]

# Create a Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print(&quot;Vocabulary: &quot;, vectorizer.vocabulary_)

# Encode the Document

vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print(&quot;Encoded Document is:&quot;)
print(vector.toarray())

