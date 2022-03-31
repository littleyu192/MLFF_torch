# system modules
import os,sys
import getopt
import subprocess

# local modules
import component.logger as mlff_logger
import parameters as pm 

# MLFF runtime option
class mlff_runtime_option:
    def __init__(self):
        # system information
        try:
            self.git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
        except:
            self.git_revision = 'UNKNOWN'
        self.commandline = ''
        # global running config
        self.force_cpu = False
        self.magic = False
        self.rseed = 2021
        self.recover_mode = False
        self.follow_mode = False
        self.shuffle_data = False
        # network config
        self.net_cfg = 'default'
        self.act = 'sigmoid'
        # optimizer config
        self.optimizer = 'ADAM'
        self.momentum = float(0)
        self.regular_wd = float(0)
        # scheduler config
        self.scheduler = 'NONE'
        self.epochs = 1000
        self.lr = float(0.1)
        self.gamma = float(0.9)
        self.step = 100
        self.batch_size = pm.batch_size
        self.dtype = pm.training_dtype
        # session/journal related config
        self.session_name = ''
        self.session_dir = ''
        self.logging_file = ''
        self.tensorboard_dir = ''
        self.model_dir = ''
        self.model_file = ''
        self.run_id = ''
        self.log_level = mlff_logger.INFO
        self.file_log_level = mlff_logger.DEBUG
        self.journal_cycle = 1
        # wandb config
        self.wandb = False
        self.wandb_entity = 'moleculenn'
        self.wandb_project = 'MLFF_torch'
        # temp config
        self.init_b = False
        self.save_model = False

        # scheduler specific options
        self.LR_milestones = None
        self.LR_patience = 0
        self.LR_cooldown = 0
        self.LR_total_steps = None
        self.LR_max_lr = 1.
        self.LR_min_lr = 0.
        self.LR_T_max = None

    def parse(self, argv):
        self.commandline = ' '.join(argv)

        opts,args = getopt.getopt(argv[1:],
            '-h-c-m-f-p-S-n:-a:-z:-v:-w:-u:-e:-l:-g:-t:-b:-d:-r:-s:-o:-i:-j:',
            ['help','cpu','magic','follow','recover','shuffle',
             'net_cfg=','act=','optimizer=','momentum=',
             'weight_decay=','scheduler=','epochs=','lr=','gamma=','step=',
             'batch_size=','dtype=','rseed=','session=','log_level=',
             'file_log_level=','j_cycle=','init_b','save_model',
             'milestones=','patience=','cooldown=','eps=','total_steps=',
             'max_lr=','min_lr=','T_max=',
             'wandb','wandb_entity=','wandb_project='])

        for opt_name,opt_value in opts:
            if opt_name in ('-h','--help'):
                print("")
                print("Help of commandline parameters for MLFF runtime option")
                print("")
                print("Generic parameters:")
                print("     -h, --help                  :  print help info")
                print("     -c, --cpu                   :  force training run on cpu")
                print("     -m, --magic                 :  a magic flag for your testing code")
                print("     -f, --follow                :  new training follow a previous trained model file")
                print("     -p, --recover               :  recover training from last breakpoint")
                print("     -S, --shuffle               :  shuffle training set during each epoch")
                print("     -n cfg, --net_cfg=cfg       :  if -f/--follow is not set, specify network cfg in parameters.py")
                print("                                    eg: -n MLFF_dmirror_cfg1")
                print("                                    if -f/--follow is set, specify the model image file name")
                print("                                    eg: '-n best1' will load model image file best1.pt from session dir")
                print("     -a act, --act=func     :  specify activation function of MLFF_dmirror")
                print("                                    current supported: [sigmoid, softplus]")
                print("     -z name, --optimizer=name   :  specify optimizer")
                print("                                    available : SGD ASGD RPROP RMSPROP ADAG")
                print("                                                ADAD ADAM ADAMW ADAMAX LBFGS")
                print("     -v val, --momentum=val      :  specify momentum parameter for optimizer")
                print("     -w val, --weight_decay=val  :  specify weight decay regularization value")
                print("     -u name, --scheduler=name   :  specify learning rate scheduler")
                print("                                    available  : LAMBDA STEP MSTEP EXP COS PLAT OC LR_INC LR_DEC")
                print("                                    LAMBDA     : lambda scheduler")
                print("                                    STEP/MSTEP : Step/MultiStep scheduler")
                print("                                    EXP/COS    : Exponential/CosineAnnealing") 
                print("                                    PLAT/OC/LR : ReduceLROnPlateau/OneCycle")
                print("                                    LR_INC     : linearly increase to max_lr")
                print("                                    LR_DEC     : linearly decrease to min_lr")
                print("     -e epochs, --epochs=epochs  :  specify training epochs")
                print("     -l lr, --lr=lr              :  specify initial training lr")
                print("     -g gamma, --gamma=gamma     :  specify gamma of StepLR scheduler")
                print("     -t step, --step=step        :  specify step_size of StepLR scheduler")
                print("     -b size, --batch_size=size  :  specify batch size")
                print("     -d dtype, --dtype=dtype     :  specify default dtype: [float64, float32]")
                print("     -r seed, --rseed=seed       :  specify random seed used in training")
                print("     -s name, --session=name     :  specify the session name, log files, tensorboards and models")
                print("                                    will be saved to subdirectory named by this session name")
                print("     -o level, --log_level=level :  specify logging level of console")
                print("                                    available: DUMP < DEBUG < SUMMARY < [INFO] < WARNING < ERROR")
                print("                                    logging msg with level >= logging_level will be displayed")
                print("     -i L, --file_log_level=L    :  specify logging level of log file")
                print("                                    available: DUMP < [DEBUG] < SUMMARY < INFO < WARNING < ERROR")
                print("                                    logging msg with level >= logging_level will be recoreded")
                print("     -j val, --j_cycle=val       :  specify journal cycle for tensorboard and data dump")
                print("                                    0: disable journaling")
                print("                                    1: record on every epoch [default]")
                print("                                    n: record on every n epochs")
                print("")
                print("scheduler specific parameters:")
                print("     --milestones=int_list       :  milestones for MultiStep scheduler")
                print("     --patience=int_val          :  patience for ReduceLROnPlateau")
                print("     --cooldown=int_val          :  cooldown for ReduceLROnPlateau")
                print("     --total_steps=int_val       :  total_steps for OneCycle scheduler")
                print("     --max_lr=float_val          :  max learning rate for OneCycle scheduler")
                print("     --min_lr=float_val          :  min learning rate for CosineAnnealing/ReduceLROnPlateau")
                print("     --T_max=int_val             :  T_max for CosineAnnealing scheduler")
                print("")
                print("wandb parameters:")
                print("     --wandb                     :  ebable wandb, sync tensorboard data to wandb")
                print("     --wandb_entity=yr_account   :  your wandb entity or account (default is: moleculenn")
                print("     --wandb_project=yr_project  :  your wandb project name (default is: MLFF_torch)")
                print("")
                exit()
            elif opt_name in ('-c','--cpu'):
                self.force_cpu = True
            elif opt_name in ('-m','--magic'):
                self.magic = True
            elif opt_name in ('-f','--follow'):
                self.follow_mode = True
                if (self.recover_mode is True):
                    raise RuntimeError("both recover_mode and follow_mode is set.")
            elif opt_name in ('-p','--recover'):
                self.recover_mode = True
                if (self.follow_mode is True):
                    raise RuntimeError("both recover_mode and follow_mode is set.")
            elif opt_name in ('-S','--shuffle'):
                 self.shuffle_data = True
            elif opt_name in ('-n','--net_cfg'):
                self.net_cfg = opt_value
            elif opt_name in ('-a','--act'):
                self.act = opt_value
            elif opt_name in ('-z','--optimizer'):
                self.optimizer = opt_value
            elif opt_name in ('-v','--momentum'):
                self.momentum = float(opt_value)
            elif opt_name in ('-w','--weight_decay'):
                self.regular_wd = float(opt_value)
            elif opt_name in ('-u','--scheduler'):
                self.scheduler = opt_value
            elif opt_name in ('-e','--epochs'):
                self.epochs = int(opt_value)
            elif opt_name in ('-l','--lr'):
                self.lr = float(opt_value)
            elif opt_name in ('-g','--gamma'):
                self.gamma = float(opt_value)
            elif opt_name in ('-t','--step'):
                self.step = int(opt_value)
            elif opt_name in ('-b','--batch_size'):
                self.batch_size = int(opt_value)
            elif opt_name in ('-d','--dtype'):
                self.dtype = opt_value
            elif opt_name in ('-r','--rseed'):
                self.rseed = int(opt_value)
            elif opt_name in ('-s','--session'):
                self.session_name = opt_value
                self.session_dir = './'+self.session_name+'/'
                self.logging_file = self.session_dir+'train.log'
                self.model_dir = self.session_dir+'model/'
                tensorboard_base_dir = self.session_dir+'tensorboard/'
                if not os.path.exists(self.session_dir):
                    os.makedirs(self.session_dir) 
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                for i in range(1000):
                    self.run_id = 'run'+str(i)
                    self.tensorboard_dir = tensorboard_base_dir+self.run_id
                    if (not os.path.exists(self.tensorboard_dir)):
                        os.makedirs(self.tensorboard_dir)
                        break
                else:
                    self.tensorboard_dir = ''
                    raise RuntimeError("reaches 1000 run dirs in %s, clean it" %self.tensorboard_dir)
            elif opt_name in ('-o','--log_level'):
                self.log_level = eval('mlff_logger.'+opt_value)
            elif opt_name in ('-i','--file_log_level'):
                self.file_log_level = eval('mlff_logger.'+opt_value)
            elif opt_name in ('-j','--j_cycle'):
                self.journal_cycle = int(opt_value)
            elif opt_name in ('--milestones'):
                self.LR_milestones = list(map(int, opt_value.split(',')))
            elif opt_name in ('--patience'):
                self.LR_patience = int(opt_value)
            elif opt_name in ('--cooldown'):
                self.LR_cooldown = int(opt_value)
            elif opt_name in ('--total_steps'):
                self.LR_total_steps = int(opt_value)
            elif opt_name in ('--max_lr'):
                self.LR_max_lr = float(opt_value)
            elif opt_name in ('--min_lr'):
                self.LR_min_lr = float(opt_value)
            elif opt_name in ('--T_max'):
                self.LR_T_max = int(opt_value)
            elif opt_name in ('--wandb'):
                self.wandb = True
                import wandb
            elif opt_name in ('--wandb_entity'):
                self.wandb_entity = opt_value
            elif opt_name in ('--wandb_project'):
                self.wandb_project = opt_value
            elif opt_name in ('--init_b'):
                self.init_b = True
            elif opt_name in ('--save_model'):
                self.save_model = True
