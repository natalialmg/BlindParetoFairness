
import sys
sys.path.append("../")

from general.utils import run_command

# dataset = 'adult'
dataset = 'lawschool'
basedir = '/data/natalia/models/'+dataset+'/BPF/'

file_bash_name = dataset+'_iw_bash.sh'

optim = 'adam' #RMSprop
model_name_prefix = 'BPF_512_adam5e6_'
# model_name_prefix = 'BPFn3_512_adam5e5_' #only have this for 512 n0 & n1
# model_name_prefix = 'BPFlr3_512_adam1e5_'

lr=5e-6
# model_name_prefix = 'BPF_64_adam1e4_'
# lr=1e-4
hlayers = '512x1'
modality = 'impweight'
batchsize=128

rho_dic = {'01':1e-1, '02': 2e-1, '03': 3e-1, '04': 4e-1, '05': 5e-1,
           '06':6e-1 ,'07':7e-1,'08':8e-1, '09':9e-1 , '1':1,
           '035':0.35, '045':0.45}

# rho_dic = {'012':1.2e-1,'014':1.4e-1,'016':1.6e-1,'018':1.8e-1}
rho_dic = {'022':2.2e-1,'024':2.4e-1,'026':2.6e-1,'028':2.8e-1}

# rho_dic = {'035':0.35, '045':0.45}
# rho_dic = {'035':0.35, '045':0.45,'015':0.15, '025':0.25, '055':0.55,
           # '065':0.65, '075':0.75, '085':0.85, '095':0.95}
# rho_dic = {'015':0.15, '025':0.25, '055':0.55, '075':0.75,  '095':0.95}

# rho_dic = {'01':1e-1, '1':1}

# epsilon_dic={'5e2':5e-2}
epsilon_dic={'1e2':1e-2}
# epsilon_dic={'0001':1e-3}
# epsilon_dic={'1e3':1e-3,'1e2':1e-2,'1e4':1e-4}
regression = False

regweight=0
loss_list = ['CE']
games=300

seed_list=[42]
split_list = [1,2,3,4,5]
split_list = [2,3,4,5]
split_list = [1]

gpu = 1

with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:
                for rho_str in rho_dic.keys():
                    for epsilon_str in epsilon_dic.keys():
                        epsilon = epsilon_dic[epsilon_str]
                        rho = rho_dic[rho_str]

                        out_file_ext = dataset + '_' + model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                       str(split) + 'rho' + rho_str + '_epsilon' + epsilon_str + '_verbose'

                        model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                     str(split) + 'rho' + rho_str + '_epsilon' + epsilon_str

                        cmd = 'python main_BPF_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"'.format(basedir, dataset, model_name)
                        # cmd = 'python main_BPF_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"  --gpu={}'.format(basedir, dataset, model_name,gpu)


                        cmd = cmd + ' --optim_regw={} --optim="{}" '.format(regweight,optim)
                        cmd = cmd + ' --rho={} --epsilon={} --lr={} --hlayers="{}" --games={} --seed={} --seed_dataset={} --split={}'.format(rho, epsilon,lr,
                                                                                                                                             hlayers, games, seed, seed, split)
                        cmd = cmd + ' --loss="{}" --regression={} --modality="{}" --batch={} > {}.txt'.format(loss, regression, modality, batchsize, out_file_ext)

                        run_command(cmd, minmem=2, use_env_variable=True, admissible_gpus=[0,1], sleep=10)
                        f.write(cmd + '\n\n\n')
                    f.write('\n\n\n')
                    f.write('\n\n\n')
                f.write('\n\n\n')