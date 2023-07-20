# Quantum-GCN pytroch implementation for ICTIR 2023 paper

## Example

1. download dataset from [google-drive](https://drive.google.com/drive/folders/1Rh9ml994jv01rBBYT276uqt5vXqR4v5a?usp=drive_link) and move those data to data directory as follows.

```
├── config/
├── dataset/
├── trainer/
├── util/
├── data/
    └── csv/
        ├── Amazon_Books.csv
        ├── gowalla.csv
        ├── ml-100k.csv
        ├── ml-1m.csv
        └── yelp.csv
```

2. execute main.py for each dataset.

```
python main.py --dataset ml-100k
python main.py --dataset ml-1m
python main.py --dataset ylep
python main.py --dataset gowalla
python main.py --dataset Amazon_Books
```

## Environment

- python == 3.7.11
- pytorch == 1.10.0
- numpy == 1.19.2
- scikit-learn == 1.0.1 
