# Mini-Leaf: A mini version of Leaf

forked from [lh-ycx/leaf](https://github.com/lh-ycx/leaf)

![Mini-Leaf](https://i.postimg.cc/CLDf3HFb/screenshot-19.png)

- What we have done

1. Simplify the code
4. Ues eager execution
5. Add ADGD and DC-ASGD algorithm
6. Modify the model (now only supports for FEMNIST) 



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
