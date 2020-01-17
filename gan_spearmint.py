import subprocess
import json
import os

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def function(batch_size, G_fil_num, D_fil_num, epochs, balanced, G_lr, D_lr, drop_rate, L1_norm_factor):

    filename = 'gan_config.json'

    data = read_json(filename)
    data['batch_size'] = batch_size[0]
    data['G_fil_num'] = G_fil_num[0]
    data['D_fil_num'] = D_fil_num[0]
    data['epochs'] = epochs[0]
    data['balanced'] = balanced[0]
    data['G_lr'] = G_lr[0]
    data['D_lr'] = D_lr[0]
    data['drop_rate'] = drop_rate[0]
    data['L1_norm_factor'] = L1_norm_factor[0]
    #data[''] = [0]

    os.remove(filename)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    command = ['/home/sq/.conda/envs/RL/bin/python', 'gan_main.py', filename]

    output = subprocess.Popen(command, stdout=subprocess.PIPE)
    val = str(output.communicate()[0])
    print(val)
    val = val[val.index('$')+1:val.index('$$')]

    return float(val)

def main(job_id, params):
    print(params)
    return function(params['batch_size'],
                    params['G_fil_num'],
                    params['D_fil_num'],
                    params['epochs'],
                    params['balanced'],
                    params['G_lr'],
                    params['D_lr'],
                    params['drop_rate'],
                    params['L1_norm_factor']
                    )
