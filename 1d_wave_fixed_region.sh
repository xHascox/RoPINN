# This is the script to obtain the results for our report, the section about Figure 5 of the original paper.

# Normal Point Optimization
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 0
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 42
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 420
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 7
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 123
python 1d_wave_point_optimization.py --model PINN --device 'cuda:0' --seed 321

python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 0
python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 42
python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 420
python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 7
python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 123
python 1d_wave_point_optimization.py --model QRes --device 'cuda:0' --seed 321


# Normal Region Optimized (with region calibration)
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 0
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 42
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 420
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 7
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 123
python 1d_wave_region_optimization.py --model PINN --device 'cuda:0' --seed 321

python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 0
python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 42
python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 420
python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 7
python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 123
python 1d_wave_region_optimization.py --model QRes --device 'cuda:0' --seed 321


# RoPINN with disabled region calibration
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 0 --initial_region 0.0000001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 42 --initial_region 0.0000001
python 1d_wave_region_optimization_noropt.py --model PINN --device 'cuda:0' --seed 420 --initial_region 0.0000001

# RoQRes with disabled region calibration
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.01
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.0001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.00001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.000001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 0 --initial_region 0.0000001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 42 --initial_region 0.0000001
python 1d_wave_region_optimization_noropt.py --model QRes --device 'cuda:0' --seed 420 --initial_region 0.0000001