## Part B - Finetuning Pretrained Model
`run.sh`is wrapper file for `train.py`. Arguments can be changed inside `run.sh`. After changing arguments, you can simply execute it. Keep `use_wandb=false` if you don't want to sweep for hyperparameters.
```
./run.sh
```
If using train.py directly, you will have to pass all the necessary arguments. \
To see usage:
```
python train.py --help
```
