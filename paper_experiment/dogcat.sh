cd ../model/

### Multi-task FL ###
echo "################################### Multi-task FL (pFed-Mul) ###################################"
echo "################################### 50-shot 10-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --cla_task_name "cla" --epoches 70 --alpha 0.05 --lamda 0.05 --init_task_weight "[[0.9, 0.1], [0.1, 0.9]]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 10 --sample_size 500 --num_agg_client 8

echo "################################### 20-shot 15-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --cla_task_name "cla" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1], [0.1, 0.9]]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 15 --sample_size 300 --num_agg_client 12

echo "################################### 10-shot 20-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --cla_task_name "cla" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1], [0.1, 0.9]]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 20 --sample_size 200 --num_agg_client 18


### Single-task classification FL ###

echo "######################### Single-task FL for classification (pFed-St) #########################"
echo "################################### 50-shot 10-client ##########################################"
python main.py --dataset "dogcat" --cla_task_name "cla" --epoches 70 --alpha 0.05 --lamda 0.05 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 10 --sample_size 500 --num_agg_client 8 --num_task 0 1

echo "################################### 20-shot 15-client ##########################################"
python main.py --dataset "dogcat" --cla_task_name "cla" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 15 --sample_size 300 --num_agg_client 12 --num_task 0 1

echo "################################### 10-shot 20-client ##########################################"
python main.py --dataset "dogcat" --cla_task_name "cla" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 20 --sample_size 200 --num_agg_client 18 --num_task 0 1


### Single-task regression FL ###

echo "######################### Single-task FL for regression (pFed-St) #########################"
echo "################################### 50-shot 10-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --epoches 70 --alpha 0.05 --lamda 0.05 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[0.1]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 10 --sample_size 500 --num_agg_client 8 --num_task 1 0

echo "################################### 20-shot 15-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[0.1]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 15 --sample_size 300 --num_agg_client 12 --num_task 1 0

echo "################################### 10-shot 20-client ##########################################"
python main.py --dataset "dogcat" --reg_task_name "reg" --epoches 50 --inner_epoches 5 --warm_up 5 --alpha 0.5 --lamda 0.1 --init_task_weight "[[0.9, 0.1]]" --init_noisereg "[0.1]" --lr_basemodel 0.0001 --lr_theta 0.001 --lr_weight 0.001 --mlp --num_client 20 --sample_size 200 --num_agg_client 18 --num_task 1 0


