import os

from elrpy.data.data import load
from elrpy.losses import lyapunov_binary_loss
from elrpy.optim import fit
from elrpy.models import init_binary

from elrpy.data.data import load

import argparse
parser = argparse.ArgumentParser(description='Pull data from voterfile/precinct results.')
parser.add_argument('-s','--state', help='State to pull voters for', type=str,required=True)
parser.add_argument('-y','--year',help='Year to pull voters for', type=int, required=True)

args = parser.parse_args()
year = args.year
state = args.state.upper()
office = "PRE"

maxit = 300

save_dir = f"data/{office.replace(' ', '_').lower()}/{state.lower()}/{year}"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

covars_path = f"{save_dir}/covars.csv.zip"
results_path = f"{save_dir}/results.csv.zip"

group_data = load(covars_path, results_path)
model_fn, model_params = init_binary(group_data)


binary_params, gd_norm = fit(
    lyapunov_binary_loss, model_fn, 
	model_params, group_data, 
    verbose=2, print_every=1, tol=1e-3,
    maxit=maxit, save_dir=save_dir
)

