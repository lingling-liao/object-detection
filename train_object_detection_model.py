import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--pipeline_config_path')
parser.add_argument('--model_dir')
args = parser.parse_args()


# Create tfrecord
p = subprocess.Popen([
    'python',
    '/tf/create_tf_record.py',
    f'--data_dir={args.data_dir}',
    '--set=train',
])
p.wait()

p = subprocess.Popen([
    'python',
    '/tf/create_tf_record.py',
    f'--data_dir={args.data_dir}',
    '--set=valid',
])
p.wait()

# Train
p = subprocess.Popen([
    'python',
    '/tf/models/research/object_detection/model_main_tf2.py',
    f'--pipeline_config_path={args.pipeline_config_path}',
    f'--model_dir={args.model_dir}',
    '--alsologtostderr',
])
p.wait()
