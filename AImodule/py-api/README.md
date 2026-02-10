## Python API

### 1) Go to the API folder

```
cd py-api
```

### 2) Create a virtual environment

```
python -m venv .venv
```

### 3) Activate it

Using Bash

```
source .venv/Scripts/activate
```

or

```
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```
pip install --upgrade pip
pip install -r requirementsenv.txt
```

### 5) Run the API

```
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
