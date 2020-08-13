import os

import pandas as pd
from pandas import CategoricalDtype
from plotnine import ggplot, aes, geom_violin, geom_boxplot, facet_grid, theme, element_text, theme_light, labs, xlab, \
    ylab, element_rect

from config import Config
from projects import ProjectName

ignore = ["clerezza", "directory-kerby", "plc4x"]
projects = list(filter(lambda x: x not in ignore, list(map(lambda x: x.github(), list(ProjectName)))))

dataset = []
scores = ['f1-measure', 'auc-roc', 'brier score']
ordered = {'auc-roc': 'AUC ROC', 'f1-measure': 'F1-Score', 'brier score': 'Brier Score'}
for project in projects:
    fowler_scores_path = Config.get_work_dir_path(os.path.join("paper", "analysis", "fowler", project, "scores.csv"))
    fowler_scores_df = pd.read_csv(fowler_scores_path)
    fowler_scores_df['dataset'] = 'Traditional'
    fowler_scores_df['project'] = project
    cond = fowler_scores_df['feature_selection'] == "all"
    fowler_scores_df = fowler_scores_df.loc[cond]
    fowler_scores_df.drop(['feature_selection', 'precision', 'recall'], axis=1, inplace=True)

    designite_fowler_scores_path = Config.get_work_dir_path(
        os.path.join("paper", "analysis", "designite_fowler", project, "scores.csv"))
    designite_fowler_scores_df = pd.read_csv(designite_fowler_scores_path)
    designite_fowler_scores_df['dataset'] = 'Designite +\n Traditional'
    designite_fowler_scores_df['project'] = project
    cond = designite_fowler_scores_df['feature_selection'] == "all"
    designite_fowler_scores_df = designite_fowler_scores_df.loc[cond]
    designite_fowler_scores_df.drop(['feature_selection', 'precision', 'recall'], axis=1, inplace=True)

    dataset += (list(map(lambda x: (
        project,
        "Traditional",
        ordered[x],
        fowler_scores_df[x].max(),
    )
                         , scores)))
    dataset += (list(map(lambda x: (
        project,
        "Designite +\nTraditional",
        ordered[x],
        designite_fowler_scores_df[x].max(),
    )
                         , scores)))

scores_order = ["AUC ROC", "Brier Score", "F1-Score"]
df = pd.DataFrame(dataset, columns=['Project', 'Smells', 'Score', 'Value'])
score_cat = pd.Categorical(df['Score'], categories=scores_order)
df = df.assign(Score=score_cat)

import matplotlib.font_manager as fm
fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
prop_text = fm.FontProperties(fname=fpath, size=20)
prop_title = fm.FontProperties(fname=fpath, size=25)
prop_strip = fm.FontProperties(fname=fpath, size=15)

(ggplot(df,
        aes(x="Smells",
            y="Value"))
 + facet_grid(['.', 'Score'])
 + geom_violin()
 + geom_boxplot(width=0.2)
 + xlab("Smells")
 + ylab("Scores")
 + theme_light()
 + theme(
            axis_text_x=element_text(fontproperties=prop_text),
            axis_text_y=element_text(fontproperties=prop_text),
            axis_title_x=element_text(fontproperties=prop_title),
            axis_title_y=element_text(fontproperties=prop_title),
            strip_background=element_rect(fill="#FAFAFA", color="#D4D4D4", size=1.5),
            strip_text=element_text(color="black", fontproperties=prop_strip)
        )
 ).save("designitetraditionalvstraditional.eps", width=50, height=20.12, units="cm")
