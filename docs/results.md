## Test accuracy

Each model is trained on one category and tested on the held-out reviews of all four. "Same category" averages the four in-category accuracies, with a 95% bootstrap interval; "other categories" averages the 12 cross-category ones.

| Model | Representation | Books | DVDs | Electronics | Kitchen | Same category | 95% CI | Other categories | Drop, points |
|---|---|---|---|---|---|---|---|---|---|
| VADER lexicon | Rule-based sentiment lexicon, cut-off chosen on the training reviews | 68.6% | 72.1% | 78.2% | 82.6% | 75.4% | 73.2% to 77.5% | 75.3% | 0.1 |
| TF-IDF | Lemmas and bigrams, negation marked, logistic regression | 83.7% | 83.9% | 91.6% | 89.6% | 87.2% | 85.7% to 88.8% | 78.9% | 8.3 |
| GloVe | Average of pretrained GloVe word vectors, logistic regression | 79.1% | 78.0% | 81.7% | 83.6% | 80.6% | 78.8% to 82.5% | 72.1% | 8.5 |
| RoBERTa embeddings | Average of frozen RoBERTa token vectors, logistic regression | 77.0% | 74.7% | 82.2% | 85.1% | 79.8% | 77.8% to 81.8% | 74.3% | 5.5 |
| Fine-tuned RoBERTa | RoBERTa fine-tuned as a classifier, top 6 of its 12 layers trained | 88.5% | 87.0% | 89.8% | 93.7% | 89.8% | 88.2% to 91.2% | 88.3% | 1.5 |

## Compared with published results

| Same category | Books | DVDs | Electronics | Kitchen | Average |
|---|---|---|---|---|---|
| Blitzer et al. (2007), linear model on unigrams and bigrams | 80.4% | 82.4% | 84.4% | 87.7% | 83.7% |
| This project: VADER lexicon | 68.6% | 72.1% | 78.2% | 82.6% | 75.4% |
| This project: TF-IDF | 83.7% | 83.9% | 91.6% | 89.6% | 87.2% |
| This project: GloVe | 79.1% | 78.0% | 81.7% | 83.6% | 80.6% |
| This project: RoBERTa embeddings | 77.0% | 74.7% | 82.2% | 85.1% | 79.8% |
| This project: Fine-tuned RoBERTa | 88.5% | 87.0% | 89.8% | 93.7% | 89.8% |

| Other categories | Average accuracy, 12 category pairs |
|---|---|
| Blitzer et al. (2007), linear model, no adaptation | 74.5% |
| Blitzer et al. (2007), SCL-MI, adapted with unlabelled target reviews | 78.0% |
| Karouzos et al. (2021), BERT-base fine-tuned on the source category | 89.9% |
| Karouzos et al. (2021), UDALM, adapted with unlabelled target reviews | 91.7% |
| This project: VADER lexicon, no adaptation | 75.3% |
| This project: TF-IDF, no adaptation | 78.9% |
| This project: GloVe, no adaptation | 72.1% |
| This project: RoBERTa embeddings, no adaptation | 74.3% |
| This project: Fine-tuned RoBERTa, no adaptation | 88.3% |

## Preprocessing experiments (TF-IDF, cross-validation on the training reviews)

| Variant | Books | DVDs | Electronics | Kitchen | Average |
|---|---|---|---|---|---|
| Words | 83.1% | 83.2% | 85.4% | 87.0% | 84.7% |
| Lemmas | 83.8% | 83.4% | 84.0% | 85.8% | 84.3% |
| Words and bigrams | 83.8% | 84.3% | 86.7% | 88.9% | 85.9% |
| Lemmas and bigrams | 84.2% | 84.3% | 87.0% | 87.7% | 85.8% |
| Words and bigrams, negation marked | 83.8% | 84.4% | 88.2% | 89.3% | 86.4% |
| Lemmas and bigrams, negation marked | 84.1% | 85.2% | 88.0% | 88.6% | 86.5% |

The TF-IDF model uses the best variant: Lemmas and bigrams, negation marked.

## Words with the strongest sentiment, per category (training reviews)

| Category | Positive | Negative |
|---|---|---|
| Books | excellent, recommend, wonderful, highly, important, entertaining, love, share, great, favorite | boring, disappointing, bad, waste, instead, skip, writing, poorly, repetitive, unfortunately |
| DVDs | enjoy, performance, intense, excellent, love, hilarious, family, great, fun, wonderful | bad, waste, boring, awful, horrible, lame, stupid, terrible, ruin, ridiculous |
| Electronics | price, great, excellent, perfect, value, good, easy, bass, highly, fast | return, refund, send, poor, waste, junk, terrible, try, stop, bad |
| Kitchen | easy, great, love, clean, perfect, size, evenly, cooking, sturdy, excellent | return, leak, disappointed, junk, waste, break, money, send, bad, poor |

## Shared sentiment vocabulary between categories (top 100 words)

| Category | Books | DVDs | Electronics | Kitchen |
|---|---|---|---|---|
| Books | 100.0% | 22.0% | 10.0% | 14.0% |
| DVDs | 22.0% | 100.0% | 11.0% | 18.0% |
| Electronics | 10.0% | 11.0% | 100.0% | 25.0% |
| Kitchen | 14.0% | 18.0% | 25.0% | 100.0% |

Correlation between shared vocabulary and cross-category accuracy over the 12 pairs: TF-IDF r = 0.82, Fine-tuned RoBERTa r = 0.52

## Text statistics

| Category | Sentiment | Reviews | Words | Sentences | Negations per 100 words | Adjectives per 100 words | Exclamation marks per 100 words |
|---|---|---|---|---|---|---|---|
| Books | negative | 963 | 186.6 | 10.7 | 1.9 | 8.4 | 0.3 |
| Books | positive | 993 | 176.8 | 10.0 | 1.2 | 8.8 | 0.3 |
| DVDs | negative | 959 | 167.6 | 10.4 | 2.0 | 8.4 | 0.5 |
| DVDs | positive | 995 | 185.0 | 10.9 | 1.3 | 8.4 | 0.6 |
| Electronics | negative | 979 | 117.9 | 8.5 | 2.5 | 7.2 | 0.6 |
| Electronics | positive | 987 | 116.6 | 8.0 | 1.7 | 8.6 | 0.5 |
| Kitchen | negative | 981 | 99.4 | 7.4 | 2.3 | 7.6 | 0.7 |
| Kitchen | positive | 996 | 98.6 | 7.2 | 1.7 | 9.4 | 1.0 |

## Most confident errors of the best model (Fine-tuned RoBERTa)

- **kitchen, positive, P(positive) = 0.00**: *It  Really Does Fit*. I was a little dubious about this item. Some of the reviews indicated that it did not really have deep pockets. Well, I can tell you that it does. My mattress is a pillow-top measuring in at 16 inches and this mattress pad accomodates it perfectly. So, do not believe the nay-saye...
- **electronics, positive, P(positive) = 0.00**: *Unbreakable*. I bought a 32 inch 5 years ago. in 2003 when movie I was trying to move it across the room and it flipped face first in the floor. Mind you I had paid like $200 for Sear's sorry 3-5 year? warranty and it turned out they wouldn't have replaced it or probably fixed it even though m...
- **books, negative, P(positive) = 1.00**: *This is a great book...*. This is a great book if you have the right teacher. The book is very informative yet if you have no back ground in abnormal psych you might want to make sure the teacher for your course is capable of explaining it to you. I also recomend that you make sure your copy has a CD rom....
- **kitchen, negative, P(positive) = 1.00**: *So So*. My kitchen is entirely Kitchenaid, so I was very excited when Kitchenaid released their silicone bakeware line. I love the color and the way they clean up very nicely. I also love that they do not get scratched or clank around in my cupboards. However, I have noticed that when I ...
- **kitchen, positive, P(positive) = 0.00**: *completly satisfied*. I couldn't be more satisfied if my stock pot had cost 10 times as much...
- **kitchen, positive, P(positive) = 0.00**: *NEVER put a carbon steel knife into the dishwasher!*. Lisa S from Ohio complains that the knives rust very easily and that this is a buyer-beware item. This is not a buyer beware item. This is an item that requires some knowledge about use and care of carbon steel knives. Carbon steel is an alloy which allows one to hone a very keen...
- **kitchen, negative, P(positive) = 0.99**: *Nowhere near as good as genuine Wusthof or Henckels...*. I recently got this set and the 2 piece carving set of the same series. I know I should not have expected much for this great price, but the knives do not have anywhere as good a feel as the classic line of wusthof or the henckels 4 or 5 star knives. They are very light in weight...
- **kitchen, positive, P(positive) = 0.01**: *Different pattern compared to usual correlle stuff*. Got damaged in shipping ...but amazon happily replaced with a new one..Great CS amazon.Thanks Had to send back the broken one back to amazon...
