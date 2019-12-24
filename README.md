# Mini-Leaf: A Simplified Benchmark for Federated Learning 


- What algorithm it supports

1. MA
2. SSGD
3. ASGD
4. DC_ASGD

![Mini-Leaf](https://i.postimg.cc/x1FLPzTH/screenshot-19.png)

- How to use

1. Generate data

```
cd mini-leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
```

2. Run

```
python main.py
```
