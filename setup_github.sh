#!/bin/bash

# GitHub Repository Setup Script
# This script helps you create a GitHub repository and push your code

echo "🚀 GitHub Repository Setup"
echo "=========================="

# Get repository name from user
read -p "Enter GitHub repository name (e.g., semantic-segmentation-platform): " repo_name

# Get GitHub username
read -p "Enter your GitHub username: " github_username

echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: $repo_name"
echo "   - Description: Semantic Segmentation Platform with Label Studio and MinIO"
echo "   - Make it Public or Private (your choice)"
echo "   - DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "   - Click 'Create repository'"
echo ""
echo "2. After creating the repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/$github_username/$repo_name.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Optional: Add a remote for SSH (if you prefer SSH):"
echo "   git remote set-url origin git@github.com:$github_username/$repo_name.git"
echo ""
echo "4. Verify the remote:"
echo "   git remote -v"
echo ""
echo "🎉 Your repository will be available at:"
echo "   https://github.com/$github_username/$repo_name"
echo ""

# Ask if user wants to run the commands now
read -p "Do you want to run the git commands now? (y/n): " run_now

if [[ $run_now == "y" || $run_now == "Y" ]]; then
    echo ""
    echo "🔗 Adding remote origin..."
    git remote add origin https://github.com/$github_username/$repo_name.git
    
    echo "🌿 Setting main branch..."
    git branch -M main
    
    echo "📤 Pushing to GitHub..."
    git push -u origin main
    
    echo ""
    echo "✅ Success! Your repository is now on GitHub:"
    echo "   https://github.com/$github_username/$repo_name"
else
    echo ""
    echo "📝 Remember to run the git commands manually after creating the repository."
fi
