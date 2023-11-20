# TODEAR


### Qucik Start

#### Entity embedding
you can run as following:
```
python model/TTransE.py 

python model/temporal.py
```

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python preprocess_data.py --data_dir data/ICEWS14
```

If you use the reward module, you need to do this step.

```
python mle_dirichlet.py --data_dir data/ICEWS14 --time_span 24
```

#### Train
```
python main.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24
```

#### Test
```
python main.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path xxxxx
```

