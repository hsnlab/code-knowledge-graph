from github import Github, RateLimitExceededException, GithubException
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import ast
import hashlib

class FunctionCollector(ast.NodeVisitor):
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.stack = []
        self.functions = {}  # name -> (start, end)

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        full_name = '.'.join(self.stack + [node.name])
        start = node.lineno - 1
        end = self._find_end_lineno(node)
        self.functions[full_name] = (start, end)
        self.generic_visit(node)

    def _find_end_lineno(self, node):
        # Try to find the maximum lineno in this function
        max_lineno = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_lineno = max(max_lineno, child.lineno)
        return max_lineno


class GitHubDataCollector:
    """
    A class that can collect github issues created/updated after a certain date and the pr-s related to them.
    """
    def __init__(self,token:str,repo_name:str, after_date:datetime):
        self.token = token
        self.repo_name = repo_name
        self.after_date = after_date
        
        self.repo = Github(self.token).get_repo(repo_name)

    def _get_issue(self, issue_number: int) -> Optional[Any]:
        """
        Get a specific issue by its number.
        """
        try:
            return self.repo.get_issue(issue_number)
        except GithubException as e:
            print(f"[Error] Could not fetch issue #{issue_number}: {e}")
            return None
        
    def _get_pull(self, pr_number: int) -> Optional[Any]:
        """
        Get a specific pull request by its number.
        """
        try:
            return self.repo.get_pull(pr_number)
        except GithubException as e:
            print(f"[Error] Could not fetch pull request #{pr_number}: {e}")
            return None

    def _hash_function_body(self,lines: List[str]) -> str:
        return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()


    def _extract_function_bodies(self,code: str) -> Dict[str, str]:
        tree = ast.parse(code)
        lines = code.splitlines(keepends=True)
        collector = FunctionCollector(lines)
        collector.visit(tree)

        func_bodies = {}
        for name, (start, end) in collector.functions.items():
            body_lines = lines[start:end]
            func_bodies[name] = self._hash_function_body(body_lines)

        return func_bodies


    def _get_changed_functions(self, base_code: str, head_code: str) -> List[str]:
        base_funcs = self._extract_function_bodies(base_code)
        head_funcs = self._extract_function_bodies(head_code)

        changed = []

        all_keys = set(base_funcs.keys()) | set(head_funcs.keys())

        for name in all_keys:
            base_hash = base_funcs.get(name)
            head_hash = head_funcs.get(name)
            if base_hash != head_hash:
                changed.append(name)

        return sorted(changed)

    def _get_changed_python_files(self,pr):
        return [
            f for f in pr.get_files()
            if f.filename.endswith(".py") and f.status in {"modified", "renamed"}
        ]

    def _get_file_content(self,repo, path, ref) -> str:
        try:
            return repo.get_contents(path, ref=ref).decoded_content.decode()
        except:
            return ""  # May happen for deleted/renamed files

    def extract_changed_functions_from_pr(self, pr) -> List[str]:
        repo = pr.base.repo
        base_sha = pr.base.sha
        head_sha = pr.head.sha

        changed_funcs = []

        for f in self._get_changed_python_files(pr):
            file_path = f.filename

            base_code = self._get_file_content(repo, file_path, base_sha)
            head_code = self._get_file_content(repo, file_path, head_sha)

            if not base_code or not head_code:
                continue

            funcs = self._get_changed_functions(base_code, head_code)
            qualified_funcs = [f"{file_path}:{fn}" for fn in funcs]
            changed_funcs.extend(qualified_funcs)

        return sorted(changed_funcs)
    
    def get_linked_prs(self,repo, issue_number):
        """
        Get all pull requests linked to or mentioning an issue
        """
        issue = self._get_issue(issue_number)#repo.get_issue(issue_number)
        linked_prs = set()
        
        for event in issue.get_timeline():
            if event.event == "cross-referenced":
                source_issue = event.source.issue
                if source_issue.pull_request and source_issue.repository_url == repo.url and source_issue.as_pull_request().state != "closed":
                    linked_prs.add(source_issue.as_pull_request())
        
        return list(linked_prs)

    def get_linked_issues(self,repo, pr_number):
        """
        Get all issue linked to or mentioning a pull request
        """
        pr = repo.get_pull(pr_number)
        linked_issues = set()
        
        for event in pr.get_timeline():
            if event.event == "cross-referenced":
                source_issue = event.source.issue
                if not source_issue.pull_request and source_issue.repository_url == repo.url:
                    linked_issues.add(source_issue)
        
        return list(linked_issues)
    
    def process_issue(self, issue_number: int) -> Dict[str, Any]:
        # Get cross-referenced prs
        try:
            linked_prs = self.get_linked_prs(self.repo, issue_number)
        except Exception as e:
            print(f"[Warning] Failed to fetch linked PRs for issue #{issue_number}: {e}")
            linked_prs = []

        if not linked_prs:
            return {}

        
        issue_data = {'linked_prs':[pr.html_url for pr in linked_prs]}
        # Get data from linked prs
        changed_funcs_all = []
        pr_problem_statements = []
        pr_comments = []
        pr_additions = 0
        pr_deletions = 0
        pr_changed_files = 0
        pr_comments_count = 0
        pr_review_comments_count = 0
        pr_commits_count = 0
        for pr in linked_prs:
            if pr.user.type == "Bot": #or "bot" in pr.user.login.lower():
                continue  # Skip bot-created PRs

            try:
                # Misc PR data
                pr_additions += pr.additions
                pr_deletions += pr.deletions
                pr_changed_files += pr.changed_files
                pr_comments_count += pr.comments
                pr_review_comments_count += pr.review_comments
                pr_commits_count += pr.commits

                # PR title and body
                pr_title = pr.title if pr.title else ""
                pr_problem_statement = pr_title + "\n" + (pr.body if pr.body else "") 
                pr_problem_statements.append(pr_problem_statement)

                # PR comments
                pr_comments_current = pr.get_issue_comments()
                pr_comments_body = "\n".join([c.body for c in pr_comments_current if c.user.type != "Bot"])
                pr_comments.append((pr_comments_body if pr_comments_body else ""))

                # Edited functions
                changed_funcs = self.extract_changed_functions_from_pr(pr)
                changed_funcs_all.extend(changed_funcs)
            except Exception as e:
                print(f"[Warning] Error processing PR {pr.html_url}: {e}")  

        issue_data['edit_functions'] = changed_funcs_all
        issue_data['pr_problem_statement'] = "\n".join(pr_problem_statements)
        issue_data['pr_comments'] = "\n".join(pr_comments)
        issue_data['pr_additions'] = pr_additions
        issue_data['pr_deletions'] = pr_deletions
        issue_data['pr_changed_files'] = pr_changed_files
        issue_data['pr_comments_count'] = pr_comments_count
        issue_data['pr_review_comments_count'] = pr_review_comments_count
        issue_data['pr_commits_count'] = pr_commits_count
        
        issue = self._get_issue(issue_number)
        if not issue:
            print(f"[Error] Could not fetch issue #{issue_number}")
            return {}
        title = issue.title if issue.title else ""
        problem_statement = title + "\n" + issue.body
        issue_data['number'] = issue.number
        issue_data['author'] = issue.user.login
        issue_data['created_at']=issue.created_at
        issue_data['closed_at']=issue.closed_at
        issue_data['state']=issue.state
        issue_data['url']=issue.html_url
        issue_data['problem_statement']=problem_statement
        issue_data['labels']= [label.name for label in issue.labels]
        # Get comments
        try:
            comments = issue.get_comments()
            issue_data['comments'] = "\n".join([c.body for c in comments if c.user.type != "Bot"])
        except Exception as e:
            print(f"[Warning] Failed to fetch comments for issue #{issue.number}: {e}")
            issue_data['comments'] = ""
        
        return issue_data
        
    
    def collect_issues(self, issue_limit:int = 100):
        """
        Download and analyze closed issues from a GitHub repository created after a specific date.
        
        Returns:
            List[Dict[str, Any]]: List of issue data
        """
        filtered_issues = []

        try:
            
            closed_issues = self.repo.get_issues(state='closed', sort='created', direction='desc', since=self.after_date)[:issue_limit]
            
            for issue in tqdm(closed_issues,desc="Fetching issues..."):
                if issue.created_at.replace(tzinfo=timezone.utc) < self.after_date.replace(tzinfo=timezone.utc):
                    break

                if issue.user.type == "Bot" or "bot" in issue.user.login.lower():
                    continue  # Skip bot-created issues
                if issue.pull_request:
                    continue
                try:                        
                        issue_data = self.process_issue(issue.number)
                        if not issue_data:
                            continue
                        else:
                            filtered_issues.append(issue_data)

                except RateLimitExceededException as e:
                    print(f"[Rate Limit] GitHub API rate limit exceeded: {e}")
                    #save_progress(filtered_issues)
                    print("You may retry after the rate limit resets.")
                    return filtered_issues
                except GithubException as e:
                    print(f"[GitHub Error] Issue #{issue.number} raised an exception: {e}")
                    continue
                except Exception as e:
                    print(f"[Error] Unexpected error with issue #{issue.number}: {e}")
                    continue
    
            return filtered_issues
        
        except RateLimitExceededException as e:
            print(f"[Fatal] Rate limit exceeded at repo level: {e}")
            #save_progress(filtered_issues)
            return filtered_issues

        except GithubException as e:
            print(f"[GitHub Error] Could not access repo {self.repo_name}: {e}")
            return []

        except Exception as e:
            print(f"[Fatal] Unexpected error initializing GitHub client: {e}")
            return []