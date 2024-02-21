from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing

dataset = datasets.Phishing()
optimizer = optim.FTRLProximal()

model = (
    preprocessing.StandardScaler() | 
    linear_model.LogisticRegression(optimizer)
)

metric = metrics.F1()
score = evaluate.progressive_val_score(dataset, model, metric)
print(score)