from github import Github
import os

# Initialize GitHub
g = Github(os.environ["GITHUB_TOKEN"])

# Get user object
user = g.get_user()

# Print username
print("Authenticated as:", user.login)

