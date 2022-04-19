source_dir=QAConv/
cd ${source_dir}


python cluster.py --data-dir Data/ --dataset cluster --save_path . --ibn b --rho 0.003 --save_data_path ../cluster_result/ --eps 0.5

