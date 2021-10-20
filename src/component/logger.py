# MLFF logging module

import logging
DUMP = 5
DEBUG = logging.DEBUG
SUMMARY = 15
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
logging.addLevelName(DUMP, 'DUMP')
logging.addLevelName(SUMMARY, 'SUMMARY')

mlff_logger = logging.getLogger('mlff')
mlff_logger.setLevel(DUMP)

def init_mlff_logger(console_log_level, file_log_level, logging_file_name):
    formatter = logging.Formatter("\33[0m\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
    handler1 = logging.StreamHandler()
    handler1.setLevel(console_log_level)
    handler1.setFormatter(formatter)
    mlff_logger.addHandler(handler1)

    if (logging_file_name != ''):
        formatter = logging.Formatter("\33[0m\33[32;49m[%(asctime)s]\33[0m.\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
        handler2 = logging.FileHandler(filename = logging_file_name)
        handler2.setLevel(file_log_level)
        handler2.setFormatter(formatter)
        mlff_logger.addHandler(handler2)

def get_module_logger(module_name):
    if (module_name == ''):
        return mlff_logger
    else:
        return logging.getLogger('mlff.'+module_name)

#
# sample to use mlff_logger in other module files
#
# # setup module logger
# import component.logger as mlff_logger
# logger = mlff_logger.get_module_logger('your_module_banner')
# def dump(msg, *args, **kwargs):
#     logger.log(mlff_logger.DUMP, msg, *args, **kwargs)
# def debug(msg, *args, **kwargs):
#     logger.debug(msg, *args, **kwargs)
# def summary(msg, *args, **kwargs):
#     logger.log(mlff_logger.SUMMARY, msg, *args, **kwargs)
# def info(msg, *args, **kwargs):
#     logger.info(msg, *args, **kwargs)
# def warning(msg, *args, **kwargs):
#     logger.warning(msg, *args, **kwargs)
# def error(msg, *args, **kwargs):
#     logger.error(msg, *args, **kwargs, exc_info=True)
#
