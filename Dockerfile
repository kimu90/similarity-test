echo "data/postgres/" >> .gitignore
git rm -r --cached data/postgres/  # Remove it from tracking if already added
git add .gitignore
git commit -m "Ignore data/postgres directory"
