import sys
sys.path.insert(0, '.')
from src.models.train import MedicineRecommender  # fixes pickle
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

r = joblib.load('models/recommender.pkl')
names = list(r.medicine_index.keys())
print('Total medicines:', len(names))
print('Threshold:', r.min_score_threshold)
print()
print('First 20 names:')
for n in names[:20]:
    print(' ', repr(n))

print()
qv = r.vectorizer.transform(['fever and headache'])
scores = cosine_similarity(qv, r.tfidf_matrix).flatten()
print('Max raw score:', round(float(scores.max()), 6))
print('Top 5 raw scores:', [round(float(s), 6) for s in sorted(scores, reverse=True)[:5]])
print('Scores above 0.10:', int((scores >= 0.10).sum()))
print('Scores above 0.01:', int((scores >= 0.01).sum()))
print()
print('Vectorizer vocab sample (first 20 terms):')
vocab = list(r.vectorizer.vocabulary_.keys())
print(' ', vocab[:20])
print()
fever_terms = [t for t in vocab if 'fever' in t or 'headache' in t]
print('Vocab terms containing fever/headache:', fever_terms[:20])