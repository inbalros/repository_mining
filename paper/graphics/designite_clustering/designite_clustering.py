import os

import pandas as pd
from matplotlib.font_manager import FontProperties
from plotnine import ggplot, aes, geom_col, geom_text, position_dodge, scale_fill_grey, theme_light, xlab, ylab, theme, \
    element_text, facet_grid, element_rect, ylim

from config import Config

designite_cluster = Config.get_work_dir_path(
    os.path.join("paper", "graphics", "designite_cluster_analysis_svm", "designite_cluster_statistics.csv"))
designite_fowler_cluster = Config.get_work_dir_path(
    os.path.join("paper", "graphics",
                 "designite_fowler_cluster_analysis_svm", "designite_cluster_statistics.csv"))
fowler_cluster = Config.get_work_dir_path(
    os.path.join("paper", "graphics",
                "fowler_cluster_analysis_svm", "designite_cluster_statistics.csv"))

designite_df = pd.read_csv(designite_cluster)
designite_fowler_df = pd.read_csv(designite_fowler_cluster)
fowler_df = pd.read_csv(fowler_cluster)

designite_df['Dataset'] = ["Designite"]*len(designite_df)
designite_fowler_df['Dataset'] = ["Designite + Traditional"]*len(designite_fowler_df)
fowler_df['Dataset'] = ["Traditional"]*len(fowler_df)

df = pd.concat([designite_df, designite_fowler_df, fowler_df], axis=0, ignore_index=True)
df['Mean'] = round(df['Mean'], 3)
fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
prop_text = FontProperties(fname=fpath, size=10)
prop_title = FontProperties(fname=fpath, size=20)
prop_strip = FontProperties(fname=fpath, size=8)

df = df.rename(columns={"Smells": "Categories"})
(ggplot(df, aes(x='Score', y="Mean", fill="Categories"))
 + geom_col(stat="identity", position="dodge")
 + facet_grid(['.', 'Dataset'])
 + scale_fill_grey()
 + theme_light()
 + xlab("Metrics")
 + ylab("Mean")
 + ylim([0, 1])
 + theme(
            axis_text=element_text(fontproperties=prop_text, rotation=30),
            axis_title=element_text(fontproperties=prop_title),
            legend_text=element_text(fontproperties=prop_text),
            legend_title=element_text(fontproperties=prop_title),
            strip_background=element_rect(fill="#FAFAFA", color="#D4D4D4", size=1.5),
            strip_text=element_text(color="black", fontproperties=prop_strip)
        )
 ).save("clusterings.eps")





