import sys
sys.path.append("../")

from general.utils import run_command

# dataset = 'adult'

dataset = 'lawschool'
basedir = '/data/natalia/models/'+dataset+'/DRO/'

file_bash_name = dataset+'_iw_bash.sh'

optim = 'adam' #RMSprop
# model_name_prefix = 'DRO_64_adam5e5_'
model_name_prefix = 'DRO_512_adam5e6_' #only have this for 512 n0 & n1

lr=5e-6

hlayers = '512x1'
modality = 'impweight'
batchsize=128

eta_dic={'0':0, '1':1, '01':0.1, '02':0.2, '03':0.3, '04':0.4,
         '05':0.5, '06':0.6,  '07':0.7, '08':0.8, '09':0.9} #eta = eta_coeff * max(eta) (e.g.: eta_coeff * np.log(2) if CE)

# eta_dic={'015':0.15, '025':0.25, '035':0.35, '045':0.45,
         # '055':0.55, '065':0.65,  '075':0.75, '085':0.85, '095':0.95}

# eta_dic={'01':0.1, '02':0.2, '04':0.4}

# eta_dic={'0':0}
eta_dic={'0':0, '1':1}
eta_dic={'01':0.1, '02':0.2, '03':0.3, '04':0.4, '05':0.5, '06':0.6,  '07':0.7, '08':0.8, '09':0.9}

regression = False
regweight=0
loss_list = ['CE']

epochs=300
seed_list=[42]
split_list = [1]
# split_list = [3,4,5]

gpu = 0

with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:
                for eta_str in eta_dic.keys():
                    eta = eta_dic[eta_str]

                    out_file_ext = dataset + '_' + model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                   str(split) + '_eta' + eta_str + '_verbose'

                    model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + \
                                 str(split) + '_eta' + eta_str

                    cmd = 'python main_DRO_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"'.format(basedir, dataset, model_name)
                    # cmd = 'python main_DRO_tabular.py --basedir="{}" --dataset="{}" --model_name="{}"  --gpu={}'.format(basedir, dataset, model_name,gpu)

                    cmd = cmd + ' --optim_regw={} --optim="{}" '.format(regweight,optim)
                    cmd = cmd + ' --eta={} --lr={} --hlayers="{}" --epochs={} --seed={} --seed_dataset={} --split={}'.format( eta, lr, hlayers, epochs, seed, seed, split)
                    cmd = cmd + ' --loss="{}" --regression={} --batch={} > {}.txt'.format(loss, regression, batchsize, out_file_ext)

                    run_command(cmd, minmem=2, use_env_variable=True, admissible_gpus=[0,1], sleep=20)
                    f.write(cmd + '\n\n\n')
                f.write('\n\n\n')
                f.write('\n\n\n')
            f.write('\n\n\n')