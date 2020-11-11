import os
import argparse
import platform
import shutil
import subprocess
import sys
import onnx
import tensorflow as tf
import onnx_tf

# Reference matrix on ONNX version, File format version, Opset versions
# https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
output_filename = 'tmp_model.pb'
report_filename = 'ModelZoo-Status.md'
verbose_mode = False
dry_run_mode = False

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
        return ''
    except Exception as e:
        first_line = str(e).strip().split('\n')[0].strip()
        return '{}: {}'.format(type(e).__name__, first_line)

def convert_model(model):
    try:
        tf_rep = onnx_tf.backend.prepare(model)
        tf_rep.export_graph(output_filename)
        del_location(output_filename)
        return ''
    except Exception as e:
        del_location(output_filename)
        strack_trace = str(e).strip().split('\n')
        if len(strack_trace) > 1:
            err_msg = strack_trace[-1].strip()
            # OpUnsupportedException gets raised as a RuntimeError
            if 'OP_UNSUPPORTED_EXCEPT' in str(e):
                err_msg = err_msg.replace(type(e).__name__,
                    'OpUnsupportedException')
            return err_msg
        else:
            return '{}: {}'.format(type(e).__name__, strack_trace[0].strip())

def capture_convert_error(model_path):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        tf_rep = onnx_tf.backend.prepare(model)
        tf_rep.export_graph(output_filename)
        del_location(output_filename)
        return ''
    except Exception as e:
        del_location(output_filename)
        return e

def report_file(models_dir, file_path, details, onnx_model_count=1):
    model_path = models_dir + '/' + file_path
    bsize = os.stat(model_path).st_size

    # - If the model file is around ~130B, it is just a file pointer.
    #   We'll download the file to test, then delete it afterwards
    #   to minimize disk utilization (important in CI environment).
    # - If you previously downloaded the file, we'll leave the
    #   the file after processing. In your local environment, make
    #   sure to pull the models you test often to avoid repetitive
    #   downloads.
    pulled = False

    if bsize <= 1024 and not dry_run_mode:
        # need to pull the model file on-demand using git lfs
        if verbose_mode:
            print('Pulling', model_path)
        cmd_args = ['git', 'lfs', 'pull', '-I', file_path, '-X', '""']
        subprocess.run(cmd_args, cwd=models_dir, stdout=subprocess.DEVNULL)
        bsize = os.stat(model_path).st_size
        pulled = True

    if bsize > 1024:
        size = size_with_units(bsize)
        model = onnx.load(model_path)
        ir_version = model.ir_version
        opset_version = model.opset_import[0].version

        check_err = '' if dry_run_mode else check_model(model)
        convert_err = '' if dry_run_mode or check_err \
            else convert_model(model)

        # https://github-emoji-list.herokuapp.com/
        if dry_run_mode:
            # skipped
            emoji_validated = ''
            emoji_converted = ''
            emoji_overall = ':heavy_minus_sign:'
        elif not check_err and not convert_err:
            # passed
            emoji_validated = ':ok:'
            emoji_converted = ':ok:'
            emoji_overall = ':heavy_check_mark:'
        elif not check_err:
            emoji_validated = ':ok:'
            emoji_converted = convert_err
            if ('BackendIsNotSupposedToImplementIt' in convert_err
                or 'OpUnsupportedException' in convert_err):
                # known limitations
                # - BackendIsNotSupposedToImplementIt: ONNX op not implemented
                # - OpUnsupportedException: TensorFlow limitation
                emoji_overall = ':warning:'
            else:
                # conversion failed
                emoji_overall = ':x:'
        else:
            # validation failed
            emoji_validated = check_err
            emoji_converted = ':heavy_minus_sign:'
            emoji_overall = ':x:'

        details.append('{} | {}. {} | {} | {} | {} | {} | {}'.format(
            emoji_overall, onnx_model_count,
            file_path[file_path.rindex('/')+1:], size, ir_version,
            opset_version, emoji_validated, emoji_converted))

        if pulled:
            # only remove model if it was pulled above on-demand
            # remove downloaded model, revert to pointer, remove cached file
            cmd_args = ('rm -f {0} && git checkout {0} && '
                'rm -f $(find . | grep $(grep oid {0} | cut -d ":" -f 2))'
                ).format(file_path)
            subprocess.call(cmd_args, cwd=models_dir, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return emoji_overall
    else:
        skipped = ':heavy_minus_sign:'
        details.append('{} | {}. {} | {} | {} | {} | {} | {}'.format(
            skipped, onnx_model_count, file_path[file_path.rindex('/')+1:],
            '', '', '', '', ''))
        return ''

# returns True if onnx models are found in the directory
def has_models(models_dir, dir_path, include=None):
    for item in os.listdir(os.path.join(models_dir, dir_path, 'model')):
        if item.endswith('.onnx'):
            file_path = os.path.join(dir_path, 'model', item)
            if include_model(file_path, include):
                return True
    return False

# file_path is relative to 'models' directory, for example:
# 'text/machine_comprehension/bert-squad'
def include_model(file_path, include):
    if include is None:
        return True
    for n in include:
        if (file_path.startswith(n)
            or file_path.endswith(n+'.onnx')
            or '/{}/model/'.format(n) in file_path):
            return True
    return False

def report_dir(models_dir, include=None,
    repo=os.getenv('GITHUB_REPOSITORY'), sha=os.getenv('GITHUB_SHA'),
    run_id=os.getenv('GITHUB_RUN_ID')):
    model_count = 0
    total_count = 0
    ok_count = 0
    warn_count = 0
    fail_count = 0
    skip_count = 0

    if repo is None:
        github_actions = 'GitHub Actions'
    else:
        # actions ([run_id](url))
        actions_url = 'https://github.com/{}/actions'.format(repo)
        github_actions = '[GitHub Actions]({})'.format(actions_url)
        if not run_id is None:
            run_link = ' ([{0}]({1}/runs/{0}))'.format(run_id, actions_url)
            github_actions += run_link
    report(':octocat: *This page is automatically generated via {}. '
        'Please do not edit manually.*'.format(github_actions))

    report('')
    report('## Environment')
    report('Package | Version')
    report('---- | -----')
    report('Platform | {}'.format(platform.platform()))
    report('Python | {}'.format(sys.version.replace('\n', ' ')))
    report('onnx | {}'.format(onnx.__version__))
    if repo is None or sha is None:
        onnxtf_version = onnx_tf.version.version
    else:
        # version ([sha](url))
        commit_url = 'https://github.com/{}/commit/{}'.format(repo, sha)
        onnxtf_version = '{} ([{}]({}))'.format(onnx_tf.version.version,
            sha[0:7], commit_url)
    report('onnx-tf | {}'.format(onnxtf_version))
    report('tensorflow | {}'.format(tf.__version__))

    # run tests first, but append to report after summary
    details = []
    for root, subdir, files in os.walk(models_dir):
        subdir.sort()
        if 'model' in subdir:
            dir_path = os.path.relpath(root, models_dir)
            if has_models(models_dir, dir_path, include):
                model_count += 1
                details.append('')
                details.append('### {}. {}'.format(model_count,
                    os.path.basename(root)))
                details.append(dir_path)
                details.append('')
                details.append('Status | Model | Size | IR | Opset | '
                    'Validated | Converted')
                details.append('------ | ----- | ---- | -- | ----- | '
                    '--------- | ---------')
        onnx_model_count = 0
        for item in sorted(files):
            if item.endswith('.onnx'):
                file_path = os.path.relpath(os.path.join(root, item),
                    models_dir)
                if include_model(file_path, include):
                    onnx_model_count += 1
                    total_count += 1
                    result = report_file(models_dir, file_path, details,
                        onnx_model_count)
                    if not result or result == ':heavy_minus_sign:':
                        skip_count += 1
                    elif result == ':heavy_check_mark:':
                        ok_count += 1
                    elif result == ':warning:':
                        warn_count += 1
                    else:
                        fail_count += 1

    report('')
    report('## Summary')
    report('Value | Count')
    report('---- | -----')
    report('Models | {}'.format(model_count))
    report('Total | {}'.format(total_count))
    report(':heavy_check_mark: Passed | {}'.format(ok_count))
    report(':warning: Limitation | {}'.format(warn_count))
    report(':x: Failed | {}'.format(fail_count))
    report(':heavy_minus_sign: Skipped | {}'.format(skip_count))

    report('')
    report('## Details')
    report('\n'.join(details))
    report('')

    print('Total: {}, Passed: {}, Limitation: {}, Failed: {}, Skipped: {}'.
        format(total_count, ok_count, warn_count, fail_count, skip_count))

def report(line):
    if verbose_mode or dry_run_mode:
        print(line)
    if not dry_run_mode:
        with open(report_filename, 'a') as f:
            f.write(line)
            f.write(os.linesep)

def del_location(loc):
    if not dry_run_mode and os.path.exists(loc):
        if os.path.isdir(loc):
            shutil.rmtree(loc)
        else:
            os.remove(loc)

def set_report_filename(output_dir, ref=os.getenv('GITHUB_REF')):
    global report_filename
    report_filename = output_dir + '/ModelZoo-Status'
    # check for a tag first
    if ref is not None:
        if '/tags/' in ref:
            report_filename += '-(tag='
        else:
            report_filename += '-(branch='
        report_filename += ref[ref.rindex('/')+1:] + ')'
    report_filename += '.md'
    del_location(report_filename)

    global output_filename
    output_filename = output_dir + '/tmp_model.pb'
    del_location(output_filename)

def generate_report(models_dir, output_dir, include=None, verbose=False,
    dry_run=False):
    if not os.path.isdir(models_dir):
        raise NotADirectoryError(models_dir)
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(output_dir)
    if subprocess.run(['git', 'lfs'], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode:
        raise EnvironmentError('git lfs is not installed')
    global verbose_mode
    verbose_mode = verbose
    global dry_run_mode
    dry_run_mode = dry_run
    set_report_filename(os.path.normpath(output_dir))
    report_dir(os.path.normpath(models_dir), include)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Test converting ONNX ModelZoo models to TensorFlow. '
        'Prerequisite: `git lfs`'))
    parser.add_argument('-m', '--models', default='models', help=(
        'ONNX ModelZoo directory (default: models)'))
    parser.add_argument('-o', '--output', default='/tmp', help=(
        'output directory (default: /tmp)'))
    parser.add_argument('-i', '--include', help=(
        'comma-separated list of models or paths to include. '
        'Use `git lfs pull` to cache frequently tested models.'))
    parser.add_argument('-v', '--verbose', action='store_true', help=(
        'verbose output'))
    parser.add_argument('--dry-run', action='store_true', help=(
        'process directory without doing conversion'))
    args = parser.parse_args()
    include = args.include.split(',') if args.include else None
    generate_report(args.models, args.output, include, args.verbose,
        args.dry_run)
