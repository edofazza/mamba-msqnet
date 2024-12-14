# Mamba-MSQNet: A Fast and Efficient Model for Animal Action Recognition


 The required Python packages can be installed with the following Bash command:

```bash
pip install -r requirements.txt
```
Then, for using correctly VideoMamba (after downloading it form the GitHub repo) you must:

```bash
pip install -e VideoMamba/causal-conv1d
pip install -e VideoMamba/mamba
```

To run the code simply fire the [`main.py`](http://main.py) file.

```bash
python mamba-MSQNet/main.py --dataset='animalkingdom' --model='videomambaclipinitvideoguidemultilayermamba' --total_length=16 --num_workers=2 --batch_size=8 --videomamba_version='m' 
```

Note: in case of using BaboonLand, change in_features (line 93, videomambaclipvideoguidemultilayermamba.py) with 29, as indicated in the comment.

