import subprocess
import json
import os

def function(batch_size, lr, epoches):

    filename = 'configuration.json'
    with open(filename, 'r') as f:
        data = json.load(f)
        data['batch_size'] = batch_size
        data['lr'] = lr
        data['epochs'] = epoches
        #data[''] =

    os.remove(filename)

    with open(filename, 'w') as f:
        json.dump(data, f)

    command = ['python3', 'main.py']

    output = subprocess.Popen(command, stdout=subprocess.PIPE)
    val = str(output.communicate()[0])
    val = val[val.index('$')+1:val.index('$$')]

    return float(val)

def main(job_id, params):
    print(params)
    return function(params['batch_size'],
                    params['lr'],
                    params['epochs'])
