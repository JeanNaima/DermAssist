# Pocket DermAssist
## ENGR 490

### TEAM 2
- Jean Naima 40210371 SOEN
- Sunil Kublalsingh 40212432 COEN
- Huy Minh Le 40209514 COEN
- Alex Bellerive 40214571 COEN
- Keith Champoux 40060023 ELEC
- Omar Selim 40155915 COEN
- Saad Khan 40177298 SOEN

### Installation
Components are tracked as **Git submodules** and must be cloned with recursion to get submodules to be cloned aswell

To clone the full project including both submodules:
```bash
git clone --recurse-submodules https://github.com/JeanNaima/DermAssist.git
```

Each component (MobileApplication) is its own Git repository.
To pull the latest changes:

```bash
git submodule update --remote --merge
```

To commit updates to the submodules’ latest commits:
```bash
git add AImodule MobileApplication
git commit -m "Update submodules to latest versions"
git push
```

## To Work with Sub Modules Generally Follow this
- `MobileApplication` is its **own full Git repository**.
- `DermAssist` only stores a **pointer** to a specific commit of `MobileApplication`.
- This means `MobileApplication` can have new commits, but `DermAssist` won’t automatically use them until you *update the submodule reference*.

# If Working on DermAssist (and integrating Mobile changes)

If you’re responsible for keeping the whole project up to date, you’ll update DermAssist’s pointer to the latest MobileApplication commit.

```bash
# Step 1: Get latest DermAssist code
cd DermAssist
git checkout <current-branch>
git pull origin <current-branch>

# Step 2: Sync submodules to recorded commits
git submodule update --init --recursive

# Step 3: Move into the submodule and pull new commits from its repo
cd MobileApplication
git checkout main
git pull origin main

# Step 4: Go back and record the new MobileApplication commit in DermAssist
cd ..
git add MobileApplication
git commit -m "Bump MobileApplication to latest main"
git push origin <current-branch>
```

This updates **DermAssist** so that it now points to the new commit in MobileApplication.

---

# If Only Working on the MobileApplication

If you’re updating the Flutter app, you work **inside** the submodule folder:

```bash
cd DermAssist/MobileApplication

# Ensure you're on the correct branch
git checkout main
git pull origin main

# Make and test changes
# (edit files, run Flutter, etc.)

git add .
git commit -m "Update camera feature"
git push origin main
```

This updates the **MobileApplication repository** only — not DermAssist.

---