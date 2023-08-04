import faulthandler

faulthandler.enable()

import os

from elrpy.data.data import load
from elrpy.losses import lyapunov_binary_loss
from elrpy.optim import get_wrapped_loss, get_mapped_fn, get_clipped_cg_fn, gd
from elrpy.utils import get_dims, get_mean_fn
from elrpy.models import init_binary

from elrpy.data.data import load

import jax
import jax.numpy as np

import argparse
parser = argparse.ArgumentParser(description='Pull data from voterfile/precinct results.')
parser.add_argument('-s','--state', help='State to pull voters for', type=str,required=True)
parser.add_argument('-y','--year',help='Year to pull voters for', type=int, required=True)

from elrpy.data.data import load

args = parser.parse_args()
year = args.year
state = args.state.upper()
office = "US POTUS"

maxit = 300

save_dir = f"data/{office.replace(' ', '_').lower()}/{state.lower()}/{year}"

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

covars_path = f"{save_dir}/covars.csv.zip"
results_path = f"{save_dir}/results.csv.zip"

group_data = load(
	covars_path, results_path,
	Y_col="votes", N_col="two_way_votes", pivot_col=["office", "party"]
)

num_groups, dim, num_outcomes = get_dims(group_data)
model_fn, model_params = init_binary(dim)

loss_fn = get_wrapped_loss(lyapunov_binary_loss, model_fn, num_groups)
loss_and_grad_fn =jax.value_and_grad(get_mean_fn(loss_fn))
hess_fn = jax.hessian(get_mean_fn(loss_fn))

mapped_loss_fn = get_mapped_fn(jax.jit(loss_fn))
mapped_loss_and_grad_fn = get_mapped_fn(jax.jit(loss_and_grad_fn))
mapped_hess_fn = get_mapped_fn(jax.jit(hess_fn))

cg_fn = get_clipped_cg_fn(mapped_loss_fn, mapped_loss_and_grad_fn, mapped_hess_fn)

binary_params, gd_norm = gd(
    lyapunov_binary_loss, model_fn, model_params, group_data, 
    verbose=2, lr=1., print_every=1, tol=1e-3,
    maxit=maxit, mapped_loss_and_dir_fn=cg_fn
)

np.savez(
	f"{save_dir}/model.npz", 
	model_params=binary_params, grad_norm=gd_norm
)

