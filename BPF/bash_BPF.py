
import sys
sys.path.append("../")

from general.utils import run_command

dataset = 'adult'
# dataset = 'lawschool'
# dataset = 'lawschool_nofaminc'
# dataset = 'compas'
basedir = '/data/natalia/models/'+dataset+'/BPF/'

file_bash_name = dataset+'_iw_bash.sh'

model_name_prefix = 'BPF_512_adam1e5_wreg1e2_'
# model_name_prefix = 'BPF_512_adam1e5_wreg1e1_'
# model_name_prefix = 'BPF_512_adam1e5_'
# model_name_prefix = 'BPF_64_adam1e5_'

## optimizer
optim = 'adam' #RMSprop
lr=1e-5
regweight = 1e-2

#Projector step
eta = 5

## Network
hlayers = '512x1'
# hlayers = '64x1'
batchsize=128

rho_dic = {'005':0.05,'01':1e-1, '02': 2e-1, '03': 3e-1, '04': 4e-1, '05': 5e-1,
           '06':6e-1 ,'07':7e-1,'08':8e-1, '09':9e-1 , '1':1,
           '035':0.35, '045':0.45,'015':0.15, '025':0.25, '055':0.55,
           '065':0.65, '075':0.75, '085':0.85, '095':0.95}
# rho_dic = {'012':1.2e-1,'014':1.4e-1,'016':1.6e-1,'018':1.8e-1}
# rho_dic = {'022':2.2e-1,'024':2.4e-1,'026':2.6e-1,'028':2.8e-1}

epsilon_dic={'1e3':1e-3,'1e2':1e-2,'1e4':1e-4,'1e1':1e-1}

epsilon_list = ['1e2','1e3']
rho_list = ['01','05','1']

epsilon_list = ['1e2','1e3']


epsilon_list = ['1e2']
rho_list = ['01','05','1']



epsilon_list = ['1e2']
rho_list = ['01','05','1']


# epsilon_list = ['1e2']

# epsilon_list = ['1e1','1e2','1e3']
# rho_list = ['01','02','03','04','05','06','07','08','09','1']


epsilon_list = ['1e2','1e3']
rho_list = ['01','02','03','04','05','06','07','08','09','1']
# rho_list = ['1','01','05']
# rho_list = ['005','01','02','03','04','06','07','08','09']


epsilon_list = ['1e3']
rho_list = ['01','05','1']


### lawschool
# epsilon_list = ['1e2','1e3']
# epsilon_list = ['1e2','1e3']

epsilon_list = ['1e2']
# rho_list = ['005','015','025','035']
rho_list = ['045','055','065','075','085']


games=300




regression = False
loss_list = ['CE']
seed_list=[42]
split_list = [1,2,3,4,5]
split_list = [1,2,3,4,5]

# split_list = [1]
train = True

gpu = 0
with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:
                for rho_str in rho_list:
                    for epsilon_str in epsilon_list:
                        epsilon = epsilon_dic[epsilon_str]
                        rho = rho_dic[rho_str]

                        out_file_ext = dataset + '_' + model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                       str(split) + 'rho' + rho_str + '_epsilon' + epsilon_str + '_verbose'

                        model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                     str(split) + 'rho' + rho_str + '_epsilon' + epsilon_str

                        cmd = 'python main_BPF_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"'.format(basedir, dataset, model_name)
                        # cmd = 'python main_BPF_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"  --gpu={}'.format(basedir, dataset, model_name,gpu)

                        cmd = cmd + ' --optim_regw={} --optim="{}" --lr={} --train={} '.format(regweight,optim,lr,train)
                        cmd = cmd + ' --rho={} --epsilon={} --hlayers="{}" --games={} --seed={} --split={}'.format(rho, epsilon,hlayers,games, seed, split)
                        cmd = cmd + ' --eta={} '.format(eta)
                        cmd = cmd + ' --loss="{}" --regression={} --batch={} > {}.txt'.format(loss, regression, batchsize, out_file_ext)

                        run_command(cmd, minmem=1.0, use_env_variable=True, admissible_gpus=[0,1], sleep=10)
                        f.write(cmd + '\n\n\n')
                    f.write('\n\n\n')
                    f.write('\n\n\n')
                f.write('\n\n\n')