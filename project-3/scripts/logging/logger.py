#!/usr/bin/env python3


# IMPORTS


import sys


# CLASS

'''
    Log things at various levels configured through constructor of logger impl.
'''

class Logger:


    '''
    CONSTRUCTOR

    args:
        - log_level: possible values are ['DEMO', 'ERROR', 'WARN', 'INFO', 'DEBUG', 'VERBOSE']
            - 'DEMO'    --> log things for demo purposes - video to show everything working at end
            - 'ERROR'   --> log errors - bad things that need to be addressed with code changes
            - 'WARN'    --> log warnings - bad things, but not necessarily errors
            - 'INFO'    --> log info - helpful things, not necessarily for debugging though
            - 'DEBUG'   --> log extra info used for debugging problems
            - 'VERBOSE' --> log extra extra info used for debugging the hardest problems

        - cumulative: boolean indicating whether logging should be cumulative
            - False means only the logs for the specified log level will be printed
    '''
    def __init__(self, log_level):
        if log_level in ['DEMO', 'ERROR', 'WARN', 'INFO', 'DEBUG', 'VERBOSE']:
            self.log_level = log_level
        else:
            raise Exception('Invalid log level: %s' % log_level)

        self.cumulative = True # global config for cumulative logging


    '''
    log the message at the specified log level

    INPUT:
        - log_level: the level to log the message at
        - log_msg: the string message to log

    OUTPUT:
        - <void> - just log (print) the given message at the given log level
    '''
    def log(self, log_level, log_msg, newline=False):
        # cumulative logging - i.e. if set to INFO level then log all DEMO, ERROR, WARN, and INFO logs
        if self.should_log(log_level):
            print(('\n' if newline else '') + '%s: %s' % (log_level, log_msg))


    def should_log(self, log_level):
        if self.cumulative:
            # cumulative logging - i.e. if set to INFO level then log all DEMO, ERROR, WARN, and INFO logs
            return self.get_log_level_numeric(log_level) <= self.get_log_level_numeric(self.log_level)
        else:
            # must be equal
            return self.get_log_level_numeric(log_level) == self.get_log_level_numeric(self.log_level)


    # convert string log level to numeric log level to allow for cumulative logging
    def get_log_level_numeric(self, log_level):
        if log_level == 'DEMO':
            return 1
        elif log_level == 'ERROR':
            return 2
        elif log_level == 'WARN':
            return 3
        elif log_level == 'INFO':
            return 4
        elif log_level == 'DEBUG':
            return 5
        elif log_level == 'VERBOSE':
            return 6
        else:
            raise Exception('Invalid log level: %s' % log_level)

        
# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running logger...')

    logger_impl = Logger('INFO')

    logger_impl.log('INFO', 'some info')
    logger_impl.log('WARN', 'some warning')
    logger_impl.log('ERROR', 'some error')
    logger_impl.log('DEMO', 'some demo')

