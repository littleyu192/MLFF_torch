'''
Author: starsparkling stars_sparkling@163.com
Date: 2022-10-10 20:14:28
LastEditors: starsparkling stars_sparkling@163.com
LastEditTime: 2023-01-07 18:59:34
FilePath: /MLFF_wu_dev/active_learning_main.py
'''
import os, sys
import json

sys.path.insert(0, "/home/wuxingxing/codespace/MLFF_wu_dev/src/pre_data")
sys.path.append("/home/wuxingxing/codespace/MLFF_wu_dev/active_learning")
sys.path.append("/home/wuxingxing/codespace/MLFF_wu_dev/src")
sys.path.append("/home/wuxingxing/codespace/MLFF_wu_dev/src/pre_data")
sys.path.append("/home/wuxingxing/codespace/MLFF_wu_dev/src/optimizer")
sys.path.append("/home/wuxingxing/codespace/MLFF_wu_dev/src/model")

from active_learning.util import make_iter_name, write_to_file
from active_learning.train_models import main_train
from active_learning.run_model_md import PWmat_MD
from active_learning.labeling import Labeling
from utils.separate_movement import MovementOp

def run_iter():
    system_info = json.load(open(sys.argv[1]))
    root_dir = system_info["work_root_path"]
    record = os.path.join(root_dir, system_info["record"])
    iter_rec = [0, -1]
    if os.path.isfile(record):
        with open (record) as frec :
            for line in frec :
                if line == '\n':
                    continue
                iter_rec = [int(x) for x in line.split()]
        print ("continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1]))

    cont = True
    ii = -1
    numb_task = 4
    max_tasks = len(system_info["iter_control"])
    
    while ii < max_tasks:#control by config.json
        ii += 1
        iter_name=make_iter_name(ii)
        print("current iter is {}".format(iter_name))
        for jj in range (numb_task) :
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] :
                continue
            task_name="task %02d"%jj
            print("{} - {}".format(iter_name, task_name))
            if   jj == 0:
                print ("training start: iter {} - task {}".format(ii, jj))
                make_train(iter_name)
            elif jj == 1:
                print ("pwmat dpkf MD start: iter {} - task {}".format(ii, jj))
                run_dpkf_md(iter_name)
            elif jj == 2:
                print ("uncertainty analyse (kpu): iter {} - task {}".format(ii, jj))
                uncertainty_analyse(iter_name) #exploring/kpu_dir
            elif jj == 3:
                print ("run_fp: iter {} - task {}".format(ii, jj))
                run_fp(iter_name)
            #record_iter
            write_to_file(record, "{} {}".format(ii, jj))

def run_fp(itername):
    lab = Labeling(itername)
    lab.do_labeling()

def run_dpkf_md(itername):
    md = PWmat_MD(itername)
    #do pwmat+dpkf md
    md.dpkf_md()
    print("{} done !".format("pwmat dpkf md_run"))
    
    #separate the MOVEMENT file to single image
    movement_path = os.path.join(md.work_dir.md_dir, "MOVEMENT")
    atom_config_save_dir = md.work_dir.md_traj_dir
    mop = MovementOp(movement_path)
    if os.path.exists(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT")) is False:
        mop.save_all_image_as_one_movement(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT"), md.out_gap)
    mop = MovementOp(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT"))
    mop.save_each_image_as_atom_config(atom_config_save_dir) # #md_traj_dir
    print("{} done !".format("movement separates to trajs"))

    md.convert2dpinput()
    print("{} done !".format("convert2dpinput"))
    
def uncertainty_analyse(itername):
    main_train(itername, "kpu")
    print("{} done !".format("calculate_kpu"))

def make_train(itername):
    main_train(itername, "train")
    print("{} done !".format("train_model"))
    
# def test():
    # cwd = os.getcwd()
    # stdpath = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/iter.0000/exploring/md_dpkf_dir"
    # os.chdir(stdpath)
    # import subprocess
    # # result = subprocess.call("bash -i gen_dpkf_data.sh", shell=True)
    # res = os.popen("bash -i gen_dpkf_data.sh")
    # # assert(result == 0)
    # print(res.readlines())


if __name__ == "__main__":
    run_iter()
    # test()
