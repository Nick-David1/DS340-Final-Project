# 🚀 GitHub Setup Instructions

Follow these steps to push your DS340 project to GitHub and collaborate with your partner.

---

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in the details:
   - **Repository name**: `ds340-sofa-consignment-ai` (or your preferred name)
   - **Description**: "Multi-modal neural network for furniture saleability prediction - DS340 Final Project"
   - **Visibility**: ⚠️ **PRIVATE** (important - contains proprietary data references)
   - **Do NOT initialize** with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

---

## Step 2: Connect Your Local Repository to GitHub

After creating the repo, GitHub will show you commands. Use these:

```bash
# Navigate to your project directory
cd "/Users/nicholasdavid/DS340 /Final Project"

# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ds340-sofa-consignment-ai.git

# Push your code
git push -u origin main
```

**Note:** Replace `YOUR_USERNAME` with your actual GitHub username!

---

## Step 3: Make Your First Commit (if not done yet)

```bash
# Check what will be committed
git status

# Commit your files
git commit -m "Initial commit: Project structure and documentation"

# Push to GitHub
git push -u origin main
```

---

## Step 4: Add Your Partner as a Collaborator

1. Go to your repository on GitHub
2. Click **"Settings"** (top right of repo page)
3. Click **"Collaborators"** in the left sidebar
4. Click **"Add people"**
5. Enter Adrian's GitHub username
6. Click **"Add [username] to this repository"**
7. Adrian will receive an email invitation to accept

---

## Step 5: Partner Setup (for Adrian)

Adrian should:

1. Accept the collaboration invitation (check email)
2. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ds340-sofa-consignment-ai.git
cd ds340-sofa-consignment-ai
```

3. Set up the Python environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Add the data files locally (NOT to git):
   - Place `sofa_data.csv` and `rejected_data.csv` in the `data/` folder
   - These will be ignored by git automatically

---

## Basic Git Workflow for Collaboration

### Before you start working:
```bash
git pull origin main
```

### After making changes:
```bash
git status                    # See what changed
git add .                     # Stage all changes
git commit -m "Description"   # Commit with message
git push origin main          # Push to GitHub
```

### Create a feature branch (recommended for larger changes):
```bash
git checkout -b feature-name
# ... make changes ...
git add .
git commit -m "Description"
git push origin feature-name
# Then create a Pull Request on GitHub
```

---

## ⚠️ Important Reminders

### DO NOT COMMIT:
- ❌ `data/sofa_data.csv`
- ❌ `data/rejected_data.csv`
- ❌ `data/images/` folder
- ❌ Model checkpoint files (`.pth`, `.pkl`)
- ❌ Virtual environment (`venv/`)

### Safe to commit:
- ✅ Code files (`.py`, `.ipynb`)
- ✅ Documentation (`.md`)
- ✅ Configuration files (`.gitignore`, `requirements.txt`)
- ✅ `data/sample_data.csv` (small anonymized sample)

**The `.gitignore` is already configured to protect you!**

---

## Troubleshooting

### If you accidentally committed sensitive data:

```bash
# Remove from git but keep locally
git rm --cached data/sofa_data.csv
git commit -m "Remove sensitive data"
git push origin main
```

### If you have merge conflicts:

```bash
git pull origin main
# Fix conflicts in your editor
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

### Check what's being tracked:

```bash
git ls-files
```

---

## Quick Reference

```bash
git status              # See current state
git log --oneline       # See commit history
git diff                # See changes
git pull                # Get latest changes
git push                # Send your changes
git checkout -b new     # Create new branch
git checkout main       # Switch to main branch
```

---

## 🎉 You're All Set!

Your project is now on GitHub and ready for collaboration. Good luck with your DS340 final project!

