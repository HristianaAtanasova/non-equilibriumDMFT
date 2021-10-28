#!/usr/bin/env python
from __future__ import print_function, division
if hasattr(__builtins__, 'raw_input'):
    input = raw_input
from numpy import *
import itertools
from os import path, makedirs, getcwd, chdir
from subprocess import call, check_output
import getpass
from datetime import timedelta
import multiprocessing as mp
from sys import argv
from time import sleep
import pickle

SLEEP_INTERVAL = 1800

if (len(argv)==1):
    exec(open('runs.param').read())
elif (len(argv)==2):
    exec(open(argv[1]).read())

cwd = getcwd()

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class JobStatus:
    def __init__(self, workdir):
        self.workdir = workdir
    def isDone(self):
        return path.exists(self.workdir+'/done')
    def isQueued(self):
        return path.exists(self.workdir+'/queued')
    def isRunning(self):
        return path.exists(self.workdir+'/running')
    def setQueued(self):
        call(r'echo " " > '+cwd+'/'+self.workdir+'/queued', shell=True)

def execute(lr):
    print('Executing '+lr.js.workdir+'.')
    lr.js.setQueued()
    call('cd '+lr.js.workdir+'; '+run_command+' runfile > jobid', shell=True)
    return 1

def executeWithQueueLimit(lr):
    jobcount = int(check_output(queue_jobcount_command, shell=True))
    while jobcount >= lr.qlimit:
        print("%d jobs in the queue, limit %d - sleeping for %d seconds." % (jobcount, lr.qlimit, SLEEP_INTERVAL))
        sleep(SLEEP_INTERVAL)
        jobcount = int(check_output(queue_jobcount_command, shell=True))
    print('Executing '+lr.js.workdir+'.')
    lr.js.setQueued()
    call('cd '+lr.js.workdir+'; '+run_command+' runfile > jobid', shell=True)
    return 1

class LocalRun:
    #define a static maximum number of concurrent processes (default 1):
    maxNumProcesses = 1
    processPool = []

    @classmethod
    def setMaxNumProcesses(cls, num):
        cls.maxNumProcesses=num

    @classmethod
    def runAll(cls):
        if len(cls.processPool)>0:
            auth = input("Run {n} jobs in queue '{queue}'? (y/n/qlimit/number to run)".format(n=str(len(cls.processPool)),queue=queueName))
            if auth == 'n' or auth == 'N':
                exit(0)
            elif auth == 'y' or auth == 'Y':
                pool = mp.Pool(processes=min(cls.maxNumProcesses,len(cls.processPool)))
                result = pool.map(execute, cls.processPool)
            elif auth == 'qlimit':
                qlimit = raw_input("Maximum number of jobs to queue simultaneously?".format(n=str(len(cls.processPool)),queue=queueName))
                cls.qlimit = int(qlimit)
                pool = mp.Pool(processes=min(cls.maxNumProcesses,len(cls.processPool)))
                result = pool.map(executeWithQueueLimit, cls.processPool)
            elif int(auth)>0 and int(auth)<len(cls.processPool):
                pool = mp.Pool(processes=min(cls.maxNumProcesses,int(auth)))
                result = pool.map(execute, cls.processPool[:int(auth)])
            else:
                print("Unrecognized directive.")
                exit(0)
        else:
            print("No jobs to run!")

    @classmethod
    def addJob(cls, job):
        cls.processPool.append(job)

    def __init__(self, js):
        self.js = js
        self.addJob(self)


#initialize a local process runner:
LocalRun.setMaxNumProcesses(numConcurrentProcesses)

#create an iterator which runs over all parameter sets:
paramSets = []
labels = []
for paramSet in params:
    labels, terms = zip(*paramSet.items())
    paramSets.append(itertools.product(*terms))
paramIterator = itertools.chain.from_iterable(paramSets)

#begin the iteration:
for term in paramIterator:
    pdict = dict(zip(labels, term))
    p = Struct(**pdict)#This makes it convenient to access the parameters as p.whatever...
    outdir = item_name(prefix, pdict, secondary_keys=secondary_keys,excluded_keys=excluded_keys)
    for valname,valfunc in post_data:
        pdict[valname] = valfunc(pdict)
    p_contents = template.format(**pdict)
    p_runfile = process_runfile(runfile, pdict, cwd+'/'+outdir)

    js = JobStatus(outdir)
    if(js.isDone()):
        pass
        #print(outdir, 'already done.')
    elif(js.isQueued()):
        print(outdir, 'already queued.')
    elif(js.isRunning()):
        print(outdir, 'already running.')
    else:
        print('Entering',outdir,'...')
        if(not path.exists(cwd+'/'+outdir)):
            makedirs(cwd+'/'+outdir)
        output = open(outdir+'/'+param_file_name, 'w')
        output.write(p_contents)
        output.close()
        output = open(outdir+'/runfile', 'w')
        output.write(p_runfile)
        output.close()

        LocalRun(js)

LocalRun.runAll()
