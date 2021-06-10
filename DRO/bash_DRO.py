import sys
sys.path.append("../")

from general.utils import run_command

dataset = 'lawschool'

# dataset = 'lawschool'
# dataset = 'lawschool_nofaminc'
# dataset = 'adult'

# dataset = 'compas'
basedir = '/data/natalia/models/'+dataset+'/DRO/'

file_bash_name = dataset+'_iw_bash.sh'

optim = 'adam' #RMSprop
# model_name_prefix = 'DRO_64_adam5e5_'
model_name_prefix = 'DRO_512_adam1e5_wreg1e2_' #only have this for 512 n0 & n1

## optimizer
optim = 'adam' #RMSprop
lr=1e-5
regweight = 1e-2

## Network
hlayers = '512x1'
batchsize=128

eta_dic={'0':0, '1':1, '01':0.1, '02':0.2, '03':0.3, '04':0.4,
         '05':0.5, '06':0.6,  '07':0.7, '08':0.8, '09':0.9} #eta = eta_coeff * max(eta) (e.g.: eta_coeff * np.log(2) if CE)


eta_dic={'0':0, '1':1,'01':0.1, '02':0.2, '03':0.3, '04':0.4, '05':0.5, '06':0.6,  '07':0.7, '08':0.8, '09':0.9,'015':0.15, '025':0.25, '035':0.35, '045':0.45,
         '055':0.55, '065':0.65,  '075':0.75, '085':0.85, '095':0.95}

# eta_list = ['0','1','05','02']
eta_list = ['0','01','02','03','04','05','06','07','08','09','1']

# eta_list = ['01','04','05','07']
# eta_list = ['06']

# eta_list = ['1','05','0']

regression = False
loss_list = ['CE']

epochs=300
seed_list=[42]
split_list = [1,2]
# split_list = [1]
split_list = [1,2,3,4,5]
train = False

gpu = 1

with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:
                for eta_str in eta_list:
                    eta = eta_dic[eta_str]

                    out_file_ext = dataset + '_' + model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                   str(split) + '_eta' + eta_str + '_verbose'

                    model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                 str(split) + '_eta' + eta_str

                    cmd = 'python main_DRO_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"'.format(basedir, dataset, model_name)
                    # cmd = 'python main_DRO_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"  --gpu={}'.format(basedir, dataset, model_name,gpu)

                    cmd = cmd + ' --optim_regw={} --optim="{}" --lr={} --train={} '.format(regweight, optim, lr,train)
                    cmd = cmd + ' --eta={} --hlayers="{}" --epochs={} --seed={} --split={}'.format( eta, hlayers, epochs, seed, split)
                    cmd = cmd + ' --loss="{}" --regression={} --batch={} > {}.txt'.format(loss, regression, batchsize, out_file_ext)

                    run_command(cmd, minmem=1.2, use_env_variable=True, admissible_gpus=[0,1], sleep=10)
                    f.write(cmd + '\n\n\n')
                f.write('\n\n\n')
                f.write('\n\n\n')
            f.write('\n\n\n')