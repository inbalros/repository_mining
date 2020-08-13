import os
from itertools import product

import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from plotnine import ggplot, aes, geom_col, scale_fill_grey, theme_light, xlab, ylab, theme, element_text, \
    geom_errorbar, position_dodge, facet_grid, geom_violin, geom_boxplot, element_rect, geom_text
from scipy.stats import sem

from config import Config
from projects import ProjectName

projects = list(map(lambda x: x.github(), ProjectName))
ignored = [
    ProjectName.Clerezza.github(),
    ProjectName.DirectoryKerby.github(),
    ProjectName.Plc4x.github()]
projects = list(filter(lambda x: x not in ignored, projects))

metrics = {'precision': 'Precision', 'recall': 'Recall', 'f1-measure': 'F1 Score', 'auc-roc': 'AUC ROC',
           'brier score': 'Brier Score'}
datasets = {'fowler_size_and_complexity': 'Size And Complexity', 'fowler_coupling': 'Coupling',
            'fowler_inheritance': 'Inheritance'}
columns = ['Project', 'Dataset', 'Metric', 'Value']
scores = []
for project in projects:
    try:
        dfs = {
            datasets[key]: pd.read_csv(
                Config.get_work_dir_path(
                    os.path.join("paper", "designite_cluster_analysis", key, project, "scores.csv")))
            for key in datasets.keys()}

        scores += list(map(lambda dataset:
                           list(map(lambda metric: (
                               project,
                               dataset,
                               metrics[metric],
                               dfs[dataset][metric].iloc[0]
                           ), metrics.keys())), dfs.keys()))
    except:
        print(project)
        pass

scores = [item for sublist in scores for item in sublist]
scores_df = pd.DataFrame(scores, columns=columns)
stats_df = scores_df.groupby(['Dataset', 'Metric']).aggregate([np.mean, sem]).reset_index()
stats_df.columns = ['Smells', 'Score', 'Mean', 'Se']
stats_df.Mean = stats_df.Mean.round(3)
stats_df.Se = stats_df.Se.round(3)
stats_df.to_csv("designite_cluster_statistics.csv", index=False)
fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
prop_text = FontProperties(fname=fpath, size=14)
prop_title = FontProperties(fname=fpath, size=20)

(ggplot(stats_df, aes(x='Score', y="Mean", fill="Smells"))
 + geom_col(stat="identity", position="dodge")
 + geom_errorbar(aes(ymin='Mean-Se', ymax='Mean+Se'),
                 width=0.2,
                 position=position_dodge(0.9))
 + scale_fill_grey()
 + theme_light()
 + xlab("Metrics")
 + ylab("Mean")
 + theme(
            axis_text=element_text(fontproperties=prop_text, rotation=20),
            axis_title=element_text(fontproperties=prop_title),
            legend_text=element_text(fontproperties=prop_text),
            legend_title=element_text(fontproperties=prop_title),
            legend_position="top"
        )
 ).save("pyramid_cluster_barplot.pdf")

(ggplot(stats_df, aes(x='Score', y="Mean", fill="Smells"))
 + geom_col(stat="identity", position="dodge")
 + geom_text(aes(x='Score', y='Mean', label='Mean'),
             position=position_dodge(width=1),
             size=5,
             va='baseline')
 + scale_fill_grey()
 + theme_light()
 + xlab("Metrics")
 + ylab("Mean")
 + theme(
            axis_text=element_text(fontproperties=prop_text, rotation=20),
            axis_title=element_text(fontproperties=prop_title),
            legend_text=element_text(fontproperties=prop_text),
            legend_title=element_text(fontproperties=prop_title),
            legend_position="top"
        )
 ).save("pyramid_cluster_barplot_values.pdf")



fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
prop_text = FontProperties(fname=fpath, size=20)
prop_title = FontProperties(fname=fpath, size=25)
prop_strip = FontProperties(fname=fpath, size=15)

(ggplot(scores_df,
        aes(x="Dataset",
            y="Value"))
 + facet_grid(['.', 'Metric'])
 + geom_violin()
 + geom_boxplot(width=0.2)
 + xlab("Smells")
 + ylab("Scores")
 + theme_light()
 + theme(
            axis_text_x=element_text(fontproperties=prop_text, rotation=90),
            axis_text_y=element_text(fontproperties=prop_text),
            axis_title_x=element_text(fontproperties=prop_title),
            axis_title_y=element_text(fontproperties=prop_title),
            strip_background=element_rect(fill="#FAFAFA", color="#D4D4D4", size=1.5),
            strip_text=element_text(color="black", fontproperties=prop_strip)
        )
 ).save("pyramid_cluster_violin_plot.pdf", width=50, height=20.12, units="cm")
