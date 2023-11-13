# TODEAR


### Qucik Start

#### Entity embedding
you can run as following:
```
python3 TTransE.py 

python3 temporal.py

```

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python3 preprocess_data.py --data_dir data/ICEWS14
```

If you use the reward module, you need to do this step.

```
python3 mle_dirichlet.py --data_dir data/ICEWS14 --time_span 24
```

#### Train
you can run as following:
```
python3 main.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24
```

#### Test
you can run as following:
```
python3 main.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path xxxxx
```

