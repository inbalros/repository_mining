import gzip
import os
import pickle

from pytest_steps import test_steps
import git

from config import Config
from data_extractor import DataExtractor
from projects import ProjectName
from repo import Repo
from version_selector import VersionType

class TestDataExtractor:

    @test_steps('git_path', 'github_name', 'jira_url',
                'jira_project_name', 'jira_repo', 'commits')
    def test__init(self):
        extractor = object.__new__(DataExtractor)
        project = ProjectName.CommonsLang.value
        extractor.git_path = project.path()
        path = os.path.join(Config().config['REPO']['RepoDir'], "commons-lang" )
        assert extractor.git_path == path
        yield
        extractor.github_name = project.github()
        assert extractor.github_name == "commons-lang"
        yield
        extractor.jira_url = Config().config['REPO']['JiraURL']
        URL = "http://issues.apache.org/jira"
        assert extractor.jira_url == URL
        yield
        extractor.jira_project_name = project.jira()
        assert extractor.jira_project_name == "LANG"
        yield
        extractor.repo = Repo(extractor.jira_project_name,
                              extractor.github_name,
                              local_path=extractor.git_path)
        path = extractor.git_path
        assert os.path.exists(path) and os.listdir(path)
        yield
        repo = git.Repo(extractor.git_path)	
        extractor.commits = extractor._get_repo_commits("", repo, extractor.jira_project_name, extractor.jira_url)	
        extractor.versions = extractor._get_repo_versions("", repo)	
        extractor.bugged_files_between_versions = extractor._get_bugged_files_between_versions()
        assert extractor.bugged_files_between_versions
        yield

    def test_extract(self):
        project = ProjectName.CommonsLang.value
        extractor = DataExtractor(project)
        extractor.extract()

    def test_bin_choose_versions(self):
        project = ProjectName.CommonsLang.value
        extractor = DataExtractor(project)
        extractor.choose_versions()
        assert True

    def test_quadratic_choose_versions(self):
        project = ProjectName.CommonsLang.value
        extractor = DataExtractor(project)
        extractor.choose_versions(algorithm="quadratic", strict="true", version_type=VersionType.Untyped)
        assert True

    def test_bugged_files_between_versions(self):
        project = ProjectName.CommonsLang.value
        extractor = DataExtractor(project)
        version = Config().config['TEST_0']['VersionName']
        files = extractor.get_files_bugged(version)
        assert files


