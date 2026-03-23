uv run src/train.py -m experiment=pino_shock_law logger=csv seed=1,2,3 data.n_train=20,30,50,70
uv run src/train.py -m experiment=pino_shock_hw logger=csv seed=1,2,3 data.n_train=20,30,50,70
uv run src/train.py -m experiment=pino_shock_pure logger=csv seed=1,2,3 data.n_train=20,30,50,70
uv run src/train.py -m experiment=pino_shock_u logger=csv seed=1,2,3 data.n_train=20,30,50,70