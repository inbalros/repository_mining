import os

import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_col, geom_errorbar, position_dodge, geom_violin, geom_boxplot
from scipy.stats import stats, sem

from config import Config
from projects import ProjectName

projects = list(map(lambda x: x.github(), list(ProjectName)))
working_projects = dict()
scores = []

datasets = {"designite": "Designite", "fowler": "Fowler",
            "designite_fowler": "Designite +\n Fowler",
            "designite_minus_fowler": "Designite -\n Fowler",
            "designite_and_fowler_minus_fowler": "Designite +\n Fowler - Fowler"
            }

metrics = {
   'precision': "Precision",
   'recall': "Recall",
   'f1-measure': "F1-Score",
   'auc-roc': "AUC ROC",
   'brier score': "Brier Score"
}
for project in projects:
    try:
        dfs = {
            datasets[key]: pd.read_csv(
                Config.get_work_dir_path(os.path.join("paper", "svm_analysis", key, project, "scores.csv")))
            for key in datasets.keys()
        }

        scores += list(map(lambda key:
                           list(map(lambda metric: (
                               project,
                               key,
                               metrics[metric],
                               round(float(dfs[key][metric].iloc[0]), 3)), metrics
                                    )),
                           dfs
                           ))

    except Exception:
        print("Failed {}".format(project))
        continue

scores = [item for sublist in scores for item in sublist]
columns = ["Project", "Dataset", "Score", "Value"]
scores_df = pd.DataFrame(scores, columns=columns)
(ggplot(scores_df, aes(x="Score", y="Value", fill="Dataset"))
 + geom_violin()
 ).save("differences_violinplot.pdf")

stats_df = scores_df.groupby(['Dataset', 'Score']).aggregate([np.mean, np.std, sem]).reset_index()
stats_df.columns = ['Dataset', 'Score', "Mean", "Sd", "Se"]
(ggplot(stats_df, aes(x="Score", y="Mean", fill="Dataset"))
    + geom_col(stat='identity', position='dodge')
    + geom_errorbar(aes(ymin='Mean-Se', ymax="Mean+Se"),
                    width=0.2,
                    position=position_dodge(0.9))
 ).save("differences_barplot.pdf")
pass
