import subprocess
import json
import os

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def function(fil_num, drop_rate, batch_size, lr, epochs, balanced):

    filename = 'cnn_config_1.5T.json'
    #filename = 'configuration.json'

    data = read_json(filename)
    data['fil_num'] = fil_num[0]
    data['drop_rate'] = drop_rate[0]
    data['batch_size'] = batch_size[0]
    data['lr'] = lr[0]
    data['epochs'] = epochs[0]
    data['balanced'] = balanced[0]
    #data[''] =

    os.remove(filename)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    command = ['/home/sq/.conda/envs/RL/bin/python', 'cnn_main.py', filename]

    output = subprocess.Popen(command, stdout=subprocess.PIPE)
    val = str(output.communicate()[0])
    print(val)
    val = val[val.index('$')+1:val.index('$$')]

    return float(val)

def main(job_id, params):
    print(params)
    return function(params['fil_num'],
                    params['drop_rate'],
                    params['batch_size'],
                    params['lr'],
                    params['epochs'],
                    params['balanced'])
