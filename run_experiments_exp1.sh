#!/bin/bash
python main.py --num_runs 5 --source_dataset RealWorld --target_dataset mhealth --data_path ../dataset --da_method ALL --exp_name closed_set
python main.py --num_runs 5 --source_dataset RealWorld --target_dataset Pamap2 --data_path ../dataset --da_method ALL --exp_name closed_set
python main.py --num_runs 5 --source_dataset Pamap2 --target_dataset RealWorld --data_path ../dataset --da_method ALL --exp_name closed_set
python main.py --num_runs 5 --source_dataset Pamap2 --target_dataset mhealth --data_path ../dataset --da_method ALL --exp_name closed_set
python main.py --num_runs 5 --source_dataset mhealth --target_dataset RealWorld --data_path ../dataset --da_method ALL --exp_name closed_set
python main.py --num_runs 5 --source_dataset mhealth --target_dataset Pamap2 --data_path ../dataset --da_method ALL --exp_name closed_set