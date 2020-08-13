import os
from itertools import product

import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_col, facet_grid

from config import Config
from projects import ProjectName

ignore = ['clerezza', "directory-kerby", "plc4x"]
projects = list(filter(lambda x: x not in ignore, list(map(lambda x: x.github(), ProjectName))))

fs = ["all", "chi2_20p", "chi2_50p",
      "mutual_info_classif_20p", "mutual_info_classif_50p",
      "f_classif_20", "f_classif_50", "recursive_elimination"]

metrics = ['f1-measure', 'auc-roc', 'brier score']
scores = {method: list() for method in fs},
for project, method in product(projects, fs):
    try:

        designite_path = Config.get_work_dir_path(os.path.join("paper", "analysis", "designite_fowler", project, "scores.csv"))
        designite_scores = pd.read_csv(designite_path)
        cond = designite_scores['feature_selection'] == method
        designite_scores = designite_scores.loc[cond]
        metrics = {metric: max(designite_scores[metric]) for metric in metrics}
        scores[0][method].append(metrics)
    except:
        pass


values = {c: np.mean(list(map(lambda x: x[c[1]], scores[0][c[0]]))) for c in list(product(fs, metrics))}

values = list(map(lambda x: x + (values[x],), values.keys()))

df = pd.DataFrame(values, columns=["Feature Selection", "Score", "Value"])
df['Value'] = df['Value'].map(lambda x: round(float(x), 3))
df.loc[df['Score'] == "brier score"].to_csv("designitefowlervsfowler_featuresselection.csv", index=False)

(ggplot(df, aes(x="Feature Selection", y="Value"))
 + geom_col()
 + facet_grid([".", "Score"])

 ).save("designitefowlervsfowler_featuresselection.pdf", width=50, height=28.12, units="cm")







