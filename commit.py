import re
import time
from datetime import datetime
try:
    from javadiff.javadiff.SourceFile import SourceFile
    from javadiff.javadiff.diff import get_commit_methods
except:
    from javadiff.SourceFile import SourceFile
    from javadiff.diff import get_commit_methods


class Commit(object):
    def __init__(self, bug_id, git_commit, issue=None, files=None):
        self._commit_id = git_commit.hexsha
        self._repo_dir = git_commit.repo.working_dir
        self._bug_id = bug_id
        if files:
            self._files = Commit.fix_renamed_files(files)
        else:
            self._files = Commit.fix_renamed_files(git_commit.stats.files.keys())
        self._methods = list()
        self._commit_date = time.mktime(git_commit.committed_datetime.timetuple())
        self._commit_formatted_date = datetime.utcfromtimestamp(self._commit_date).strftime('%Y-%m-%d %H:%M:%S')
        self.issue = issue

    def is_bug(self):
        return self._bug_id != '0'

    def get_issue_url(self):
        if self.issue:
            return self.issue.url
        return ""

    def get_commit_methods(self):
        if self.is_bug():
            if len(self._methods) == 0:
                self._methods = get_commit_methods(self._repo_dir, self._commit_id, analyze_source_lines=False)
        return self._methods

    @classmethod
    def init_commit_by_git_commit(cls, git_commit, bug_id=0, issue=None, files=None):
        return Commit(bug_id, git_commit, issue, files=files)

    def to_list(self):
        return [self._commit_id, str(self._bug_id), ";".join(self._files)]

    @staticmethod
    def fix_renamed_files(files):
        """
        fix the paths of renamed files.
        before : u'tika-core/src/test/resources/{org/apache/tika/fork => test-documents}/embedded_with_npe.xml'
        after:
        u'tika-core/src/test/resources/org/apache/tika/fork/embedded_with_npe.xml'
        u'tika-core/src/test/resources/test-documents/embedded_with_npe.xml'
        :param files: self._files
        :return: list of modified files in commit
        """
        new_files = []
        for file in files:
            if "=>" in file:
                if "{" and "}" in file:
                    # file moved
                    src, dst = file.split("{")[1].split("}")[0].split("=>")
                    fix = lambda repl: re.sub(r"{[\.a-zA-Z_/\-0-9]* => [\.a-zA-Z_/\-0-9]*}", repl.strip(), file)
                    new_files.extend(map(fix, [src, dst]))
                else:
                    # full path changed
                    new_files.extend(map(lambda x: x.strip(), file.split("=>")))
                    pass
            else:
                new_files.append(file)
        return new_files
