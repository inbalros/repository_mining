import os
from itertools import product

import pandas as pd
import numpy as np

from config import Config
from projects import ProjectName

projects = list(map(lambda x: x.github(), ProjectName))

metrics = ['precision', 'recall', 'f1-measure', 'auc-roc', 'brier score']
scores = {'Traditional':
              {'Mean': list(), 'Max': list()},
          'Designite':
              {'Mean': list(), 'Max': list()},
          }
for project in projects:
    if project == "clerezza" or project == "directory-kerby" or project == "plc4x":
        continue
    try:
        fowler_path = Config.get_work_dir_path(os.path.join("paper", "analysis", "fowler", project, "scores.csv"))
        fowler_scores = pd.read_csv(fowler_path)
        cond = fowler_scores['feature_selection'] == 'all'
        fowler_scores = fowler_scores.loc[cond]
        average_metrics = {metric: np.mean(fowler_scores[metric]) for metric in metrics}
        max_metrics = {metric: max(fowler_scores[metric]) for metric in metrics}
        scores['Traditional']['Mean'].append(average_metrics)
        scores['Traditional']['Max'].append(max_metrics)

        designite_path = Config.get_work_dir_path(os.path.join("paper", "analysis", "designite_fowler", project, "scores.csv"))
        designite_scores = pd.read_csv(designite_path)
        cond = designite_scores['feature_selection'] == 'all'
        designite_scores = designite_scores.loc[cond]
        average_metrics = {metric: np.mean(designite_scores[metric]) for metric in metrics}
        max_metrics = {metric: max(designite_scores[metric]) for metric in metrics}
        scores['Designite']['Mean'].append(average_metrics)
        scores['Designite']['Max'].append(max_metrics)
    except:
        print(project)
        pass

datasets = ['Traditional', 'Designite']
criterion = ['Mean', 'Max']

values = {c: np.std(list(map(lambda x: x[c[2]], scores[c[0]][c[1]]))) for c in list(product(datasets, criterion, metrics))}

values = list(map(lambda x: x + (values[x],), values.keys()))

df = pd.DataFrame(values, columns=["Smells", "Classifier Selection", "Score", "Value"])
df['Value'] = df['Value'].map(lambda x: round(float(x), 3))
df.to_csv("designitefowlervsfowler.csv", index=False)







