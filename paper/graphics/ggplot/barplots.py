import pandas as pd
from matplotlib.font_manager import FontProperties
from plotnine import ggplot, aes, geom_col, theme_light, scale_fill_grey, theme, element_text, xlab, ylab, geom_text, \
    position_dodge

smells_order = ["Designite", "Traditional", "Designite + Traditional"]
scores_order = ["Precision", "Recall", "F1-Score", "AUC-ROC", "Brier Score"]

values = [("Designite", "Precision", 0.242),
          ("Designite", "Recall", 0.501),
          ("Designite", "F1-Score", 0.269),
          ("Designite", "AUC-ROC", 0.583),
          ("Designite", "Brier Score", 0.382),
          ("Traditional", "Precision", 0.241),
          ("Traditional", "Recall", 0.727),
          ("Traditional", "F1-Score", 0.319),
          ("Traditional", "AUC-ROC", 0.697),
          ("Traditional", "Brier Score", 0.350),
          ("Designite + Traditional", "Precision", 0.270),
          ("Designite + Traditional", "Recall", 0.761),
          ("Designite + Traditional", "F1-Score", 0.340),
          ("Designite + Traditional", "AUC-ROC", 0.738),
          ("Designite + Traditional", "Brier Score", 0.304)]

fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
prop_text = FontProperties(fname=fpath, size=14)
prop_title = FontProperties(fname=fpath, size=20)

df = pd.DataFrame(values, columns=['Smells', 'Score', 'Value'])
# smells_cat = pd.Categorical(df['Smells'], categories=smells_order)
# scores_cat = pd.Categorical(df['Score'], categories=scores_order)
# df = df.assign(Smells=smells_cat)
# df = df.assign(Score=scores_cat)


(ggplot(df, aes(x='Score', y="Value", fill="Smells"))
 + geom_col(stat="identity", position="dodge")
 + geom_text(aes(x='Score', y='Value', label='Value'),
             size=5,
             va='baseline',
             position=position_dodge(0.9)
             )
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
 ).save("compareallbarplot.eps")