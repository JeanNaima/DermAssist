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

