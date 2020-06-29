This repository contains code for [this blog post]() 

# Usage

To get the help message simply run:

```bash
python main.py --help
```

Before running the script, make sure to build the mlflow docker image and to start it:

```bash
docker build . -t mlflow:local
docker run --rm -d -p 5000:5000 mlflow:local
```