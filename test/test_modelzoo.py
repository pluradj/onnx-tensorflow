import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

#logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('tensorflow').disabled

import shutil
import time
import onnx
import tensorflow as tf
import onnx_tf
from onnx_tf import version

# https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions

output_filename = '/tmp/model.pb'
report_filename = '/tmp/report.md'

def size_with_units(size):
    if size < 1024:
        s = '{}B'.format(size)
    elif size < (1024 * 1024):
        s = '{}K'.format(round(size/1024))
    elif size < (1024 * 1024 * 1024):
        s = '{}M'.format(round(size/(1024*1024)))
    else:
        s = '{}G'.format(round(size/(1024*1024*1024)))
    return s

def check_model(model):
    try:
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        first_line = str(e).strip().split('\n')[0].strip()
        return '{}: {}'.format(type(e).__name__, first_line)

def convert_model(model):
    try:
        tf_rep = onnx_tf.backend.prepare(model)
        model_pb = output_filename
        if os.path.exists(model_pb):
            if os.path.isdir(model_pb):
                shutil.rmtree(model_pb)
            else:
                os.remove(model_pb)
        tf_rep.export_graph(model_pb)
        return True
    except Exception as e:
        last_line = str(e).strip().split('\n')
        if len(last_line) > 1:
            return last_line[-1].strip()
        else:
            return '{}: {}'.format(type(e).__name__, last_line[0].strip())

def capture_convert_error(file_path):
    try:
        model = onnx.load(file_path)
        onnx.checker.check_model(model)
        tf_rep = onnx_tf.backend.prepare(model)
        model_pb = output_filename
        if os.path.exists(model_pb):
            if os.path.isdir(model_pb):
                shutil.rmtree(model_pb)
            else:
                os.remove(model_pb)
        tf_rep.export_graph(model_pb)
        return True
    except Exception as e:
        return e

def report_file(file_path, onnx_model_count=1, do_conversion=True):
    bsize = os.stat(file_path).st_size
    if bsize > 1024:
        size = size_with_units(bsize)
        model = onnx.load(file_path)
        ir_version = model.ir_version
        opset_version = model.opset_import[0].version
        checked = check_model(model)
        if do_conversion is False:
            converted = ''
        elif isinstance(checked, str) or checked is False:
            converted = False
        else:
            converted = convert_model(model)
        # https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md
        emoji = ':ok:' if checked is True and (do_conversion is False or
            converted is True) else ':x:'
        report('{} | {} {} | {} | {} | {} | {} | {}'.format(emoji, onnx_model_count,
            file_path, size, ir_version, opset_version, checked, converted))

def report_dir(dir_path, do_conversion=True):
    model_count = 0
    total_count = 0
    report('Status | Model | Size | IR | Opset | Validated | Converted')
    report('------ | ----- | ---- | -- | ----- | --------- | ---------')
    for root, subdir, files in os.walk(dir_path):
        subdir.sort()
        if 'model' in subdir:
            model_count = model_count + 1
            report('{} | {}'.format(model_count, root))
        onnx_model_count = 0
        for item in sorted(files):
            if item.endswith(".onnx") :
                onnx_model_count = onnx_model_count + 1
                total_count = total_count + 1
                file_path = str(os.path.join(root, item))
                report_file(file_path, onnx_model_count, do_conversion)
    onnx_version = onnx.__version__
    onnx_tf_version = onnx_tf.version.version
    tf_version = tf.__version__
    report('')
    report('Name | Value')
    report('---- | -----')
    report('Model count | {}'.format(model_count))
    report('Total count | {}'.format(total_count))
    report('ONNX version | {}'.format(onnx_version))
    report('ONNX-TF version | {}'.format(onnx_tf_version))
    report('TF version | {}'.format(tf_version))

def report(line):
    print(line)
    if not report_filename is None:
        with open(report_filename, 'a') as f:
            f.write(line)
            f.write(os.linesep)

def del_report():
    if os.path.exists(report_filename):
        os.remove(report_filename)

def generate_report(dir_path='./'):
    del_report()
    t0 = time.time()
    print(time.asctime(time.localtime(t0)))
    report_dir(dir_path)
    t1 = time.time()
    print(time.asctime(time.localtime(t1)))
    print(round((t1-t0)/60, 1), 'mins')

if __name__ == "__main__":
    generate_report()
