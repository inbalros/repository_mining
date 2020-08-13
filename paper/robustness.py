import itertools
import os
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from plotnine import ggplot, aes, geom_col, lims, geom_text, position_dodge, facet_grid, theme, element_text, \
    scale_fill_grey, theme_light, geom_bar, ylab

from config import Config
from projects import ProjectName


class RobustnessBase(ABC):

    # The oracle should be the best result from fowler + Designite together and fowler or designite.
    def __init__(self, base, dataset):
        self.base_path = Config.get_work_dir_path(os.path.join("paper", base))
        self.path = os.path.join(self.base_path, dataset)
        oracles = self.get_oracles()
        dfs = self.get_dfs()
        self.robustness = self.get_robustness_df(dfs, oracles)

    def get_dfs(self):
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        projects = list(filter(lambda x: x not in ignore, projects))
        dfs = {project: pd.read_csv(os.path.join(self.path, project, "scores.csv")) for project in projects}
        return dfs

    @staticmethod
    @abstractmethod
    def get_oracles():
        pass

    @staticmethod
    @abstractmethod
    def get_robustness_df(dfs, oracles):
        pass

    def create_graphs(self):
        self.robustness['Avg(Error)'] = round(self.robustness['Avg(Error)'], 3)
        self.robustness['Max(Error)'] = round(self.robustness['Max(Error)'], 3)
        dir_path = Config.get_work_dir_path(os.path.join("paper", "graphics", "robustness"))
        Config.assert_dir_exists(dir_path)
        fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
        prop_text = FontProperties(fname=fpath, size=10)
        prop_title = FontProperties(fname=fpath, size=14)
        (ggplot(self.robustness, aes(x="Score", y="Max(Error)", fill="Feature Selection"))
         + geom_col(stat="identity", position="dodge")
         + geom_text(aes(x="Score", y="Max(Error)", label="Max(Error)"),
                     position=position_dodge(width=1),
                     size=3,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "max_error_bar_grouped.pdf"))
        (ggplot(self.robustness, aes(x="Score", y="Avg(Error)", fill="Feature Selection"))
         + geom_col(stat="identity", position="dodge")
         # + lims(y=(-0.05, 0.05))
         + geom_text(aes(x="Score", y="Avg(Error)", label="Avg(Error)"),
                     position=position_dodge(width=1),
                     size=3,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "avg_error_bar_grouped.pdf"))
        (ggplot(self.robustness, aes(x="Feature Selection", y="Max(Error)", fill="Feature Selection"))
         + geom_col()
         + facet_grid(['.', 'Score'])
         + geom_text(aes(x="Feature Selection", y="Max(Error)", label="Max(Error)"),
                     size=3,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_text_x=element_text(angle=90),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "max_error_bar_facets.pdf"))
        (ggplot(self.robustness, aes(x="Feature Selection", y="Avg(Error)", fill="Feature Selection"))
         + geom_col()
         + facet_grid(['.', 'Score'])
         + geom_text(aes(x="Feature Selection", y="Avg(Error)", label="Avg(Error)"),
                     size=3,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_text_x=element_text(angle=90),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "avg_error_bar_facets.pdf"))


class RobustnessBestAlgorithm(RobustnessBase):
    def __init__(self):
        super().__init__(base="analysis", dataset="designite_fowler")

    @staticmethod
    def get_oracles():
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        projects = list(filter(lambda x: x not in ignore, projects))
        datasets = ["fowler", "designite", "designite_fowler"]
        paths = [Config.get_work_dir_path(os.path.join("paper", "analysis", dataset)) for dataset in datasets]

        def choose_max(project, paths):
            scores = ["precision", "recall", "f1-measure", "auc-roc", "brier score"]
            dfs = [pd.read_csv(os.path.join(path, project, "scores.csv")) for path in paths]
            return {score: pd.concat(dfs)[score].max() for score in scores}

        oracles = {project: choose_max(project, paths) for project in projects}
        return oracles

    @staticmethod
    def get_robustness_df(dfs, oracles):
        scores = ['precision', 'recall', 'f1-measure', 'auc-roc', 'brier score']
        feature_selections = ["chi2_20p", "chi2_50p", "mutual_info_classif_20p", "mutual_info_classif_50p",
                              "f_classif_20", "f_classif_50", "recursive_elimination"]
        scores = list(map(lambda item:
                          (
                              item[0],  # project
                              item[1],  # score
                              item[2],  # feature_selection
                              oracles[item[0]][item[1]] -
                              dfs[item[0]].loc[dfs[item[0]]['feature_selection'] == item[2]][item[1]].max()
                          ),
                          itertools.product(dfs.keys(), scores, feature_selections)))
        columns = ['Project', 'Score', 'Feature Selection', 'Error']
        scores_df = pd.DataFrame(scores, columns=columns)
        robustness_df = scores_df.groupby(['Score', 'Feature Selection']).aggregate([np.mean, max]).reset_index()
        robustness_df.columns = ['Score', 'Feature Selection', 'Avg(Error)', 'Max(Error)']
        return robustness_df


class RobustnessFSSVM(RobustnessBase):
    def __init__(self):
        super().__init__(base="svm_fs_analysis", dataset="designite_fowler")

    @staticmethod
    def get_oracles():
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        projects = list(filter(lambda x: x not in ignore, projects))
        datasets = ["fowler", "designite", "designite_fowler"]
        paths = [Config.get_work_dir_path(os.path.join("paper", "svm_analysis", dataset)) for dataset in datasets]

        def choose_max(project, paths):
            scores = ["precision", "recall", "f1-measure", "auc-roc", "brier score"]
            dfs = [pd.read_csv(os.path.join(path, project, "scores.csv")) for path in paths]
            return {score: pd.concat(dfs)[score].max() for score in scores}

        oracles = {project: choose_max(project, paths) for project in projects}
        return oracles

    @staticmethod
    def get_robustness_df(dfs, oracles):
        scores = ['precision', 'recall', 'f1-measure', 'auc-roc', 'brier score']
        feature_selections = ["chi2_20p", "chi2_50p", "mutual_info_classif_20p", "mutual_info_classif_50p",
                              "f_classif_20", "f_classif_50", "recursive_elimination"]
        scores = list(map(lambda item:
                          (
                              item[0],  # project
                              item[1],  # score
                              item[2],  # feature_selection
                              oracles[item[0]][item[1]] -
                              dfs[item[0]].loc[dfs[item[0]]['feature_selection'] == item[2]][item[1]].values[0]
                          ),
                          itertools.product(dfs.keys(), scores, feature_selections)))
        columns = ['Project', 'Score', 'Feature Selection', 'Error']
        scores_df = pd.DataFrame(scores, columns=columns)
        robustness_df = scores_df.groupby(['Score', 'Feature Selection']).aggregate([np.mean, max]).reset_index()
        robustness_df.columns = ['Score', 'Feature Selection', 'Avg(Error)', 'Max(Error)']
        return robustness_df


class RobustnessSVM(RobustnessBase):
    def __init__(self):
        super().__init__(base="svm_analysis", dataset="designite_fowler")

    def get_dfs(self):
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        datasets = {'fowler': 'Traditional', 'designite': 'Designite', 'designite_fowler': 'Designite + Traditional'}
        projects = list(filter(lambda x: x not in ignore, projects))
        dfs = {
            project: {datasets[dataset]: pd.read_csv(os.path.join(self.base_path, dataset, project, "scores.csv")) for
                      dataset in datasets.keys()} for project in projects}
        return dfs

    @staticmethod
    def get_oracles():
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        projects = list(filter(lambda x: x not in ignore, projects))
        datasets = ["fowler", "designite", "designite_fowler"]
        paths = [Config.get_work_dir_path(os.path.join("paper", "svm_analysis", dataset)) for dataset in datasets]

        def choose_max(project, ps):
            scores = ["precision", "recall", "f1-measure", "auc-roc", "brier score"]
            dfs = [pd.read_csv(os.path.join(path, project, "scores.csv")) for path in ps]
            return {score: pd.concat(dfs)[score].max() for score in scores}

        oracles = {project: choose_max(project, paths) for project in projects}
        return oracles
        pass

    @staticmethod
    def get_robustness_df(dfs, oracles):
        scores = ['precision', 'recall', 'f1-measure', 'auc-roc', 'brier score']
        datasets = ['Traditional', 'Designite', 'Designite + Traditional']
        scores = list(map(lambda item:
                          (
                              item[0],  # project
                              item[2],  # dataset
                              item[1],  # score
                              oracles[item[0]][item[1]] -
                              dfs[item[0]][item[2]].loc[dfs[item[0]][item[2]]['feature_selection'] == 'all'][
                                  item[1]].values[0]
                          ),
                          itertools.product(dfs.keys(), scores, datasets)))
        columns = ['Project', 'Smells', 'Score', 'Error']
        scores_df = pd.DataFrame(scores, columns=columns)
        robustness_df = scores_df.groupby(['Smells', 'Score']).aggregate([np.mean, max]).reset_index()
        robustness_df.columns = ['Smells', 'Score', 'Avg(Error)', 'Max(Error)']
        return robustness_df

    def create_graphs(self):
        self.robustness['Avg(Error)'] = round(self.robustness['Avg(Error)'], 3)
        self.robustness['Max(Error)'] = round(self.robustness['Max(Error)'], 3)
        dir_path = Config.get_work_dir_path(os.path.join("paper", "graphics", "robustness"))
        Config.assert_dir_exists(dir_path)
        fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
        prop_text = FontProperties(fname=fpath, size=10)
        prop_title = FontProperties(fname=fpath, size=14)
        (ggplot(self.robustness, aes(x="Score", y="Max(Error)", fill="Smells"))
         + geom_col(stat="identity", position="dodge")
         + geom_text(aes(x="Score", y="Max(Error)", label="Max(Error)"),
                     position=position_dodge(width=1),
                     size=5,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "max_error_bar_grouped.eps"))
        (ggplot(self.robustness, aes(x="Score", y="Avg(Error)", fill="Smells"))
         + geom_col(stat="identity", position="dodge")
         # + lims(y=(-0.05, 0.05))
         + geom_text(aes(x="Score", y="Avg(Error)", label="Avg(Error)"),
                     position=position_dodge(width=1),
                     size=5,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "avg_error_bar_grouped.eps"))
        (ggplot(self.robustness, aes(x="Smells", y="Max(Error)", fill="Score"))
         + geom_col()
         + facet_grid(['.', 'Score'])
         + geom_text(aes(x="Smells", y="Max(Error)", label="Max(Error)"),
                     size=5,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_text_x=element_text(angle=90),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "max_error_bar_facets.eps"))
        (ggplot(self.robustness, aes(x="Smells", y="Avg(Error)", fill="Score"))
         + geom_col()
         + facet_grid(['.', 'Score'])
         + geom_text(aes(x="Smells", y="Avg(Error)", label="Avg(Error)"),
                     size=5,
                     va='baseline')
         + scale_fill_grey()
         + theme_light()
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_text_x=element_text(angle=90),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "avg_error_bar_facets.eps"))


class RobustnessSVMDistribution:
    def __init__(self):
        self.projects = self.get_projects()
        self.datasets = ["fowler", "designite", "designite_fowler"]
        self.base_path = Config.get_work_dir_path(os.path.join("paper", "svm_analysis"))
        self.dfs = self.get_dfs()
        self.distributions = self.get_distributions()
        return

    @staticmethod
    def get_projects():
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        ignore = ["clerezza", "directory-kerby", "plc4x"]
        projects = list(filter(lambda x: x not in ignore, projects))
        return projects

    def get_dfs(self):
        dfs = list(map(lambda item:
                       (
                           item[0],
                           item[1],
                           pd.read_csv(
                               os.path.join(self.base_path, item[1], item[0], "scores.csv"))
                       ),
                       itertools.product(self.projects, self.datasets)))

        columns = ['Project', 'Dataset', 'Precision', 'Recall', 'F1 Measure', 'AUC-ROC', 'Brier Score']
        return pd.DataFrame(list(map(lambda item:
                                     (
                                         item[0],
                                         item[1],
                                         item[2]['precision'].iloc[0],
                                         item[2]['recall'].iloc[0],
                                         item[2]['f1-measure'].iloc[0],
                                         item[2]['auc-roc'].iloc[0],
                                         item[2]['brier score'].iloc[0]
                                     ), dfs)), columns=columns)

    def get_distributions(self):
        scores = ['Precision', 'Recall', 'F1 Measure', 'AUC-ROC', 'Brier Score']

        def get_best_dataset(dfs, project, score):
            elements = dfs.loc[dfs['Project'] == project][score]
            if score == "Brier Score":
                m = elements.min()
            else:
                m = elements.max()
            index = [i for i, j in enumerate(elements) if j == m]
            return dfs.loc[dfs['Project'] == project]['Dataset'].values[index][0]

        dists = list(map(lambda item:
                         (
                             item[0],
                             item[1],
                             get_best_dataset(self.dfs, item[0], item[1])
                         )
                         , itertools.product(self.projects, scores)))
        dist_dfs = pd.DataFrame(dists, columns=["Project", "Score", "Smell"])
        dist_dfs.replace(
            {'fowler': 'Traditional', 'designite': 'Designite', 'designite_fowler': 'Designite + Traditional'},
            inplace=True)
        return dist_dfs

    def get_graphics(self):
        dir_path = Config.get_work_dir_path(os.path.join("paper", "graphics", "robustness"))
        fpath = '/Users/brunomachado/Library/Fonts/cmunrm.ttf'
        prop_text = FontProperties(fname=fpath, size=10)
        prop_title = FontProperties(fname=fpath, size=14)
        (ggplot(self.distributions)
         + geom_bar(aes(x="Score", fill="Smell"))
         + theme_light()
         + scale_fill_grey()
         + ylab("% of Projects")
         + theme(
                    axis_text=element_text(fontproperties=prop_text, rotation=30),
                    axis_title=element_text(fontproperties=prop_title),
                    legend_text=element_text(fontproperties=prop_text),
                    legend_title=element_text(fontproperties=prop_title))
         ).save(os.path.join(dir_path, "robustnessdistribution.eps"))


if __name__ == "__main__":
    # RobustnessBestAlgorithm().create_graphs()
    RobustnessSVM().create_graphs()
    # RobustnessFSSVM().create_graphs()
    # RobustnessSVMDistribution().get_graphics()
