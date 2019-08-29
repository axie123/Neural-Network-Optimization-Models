#!/usr/bin/env python 
# Originally written by Dr. Pascal Frederich. I modified it to fit my neural network model.
# From the Aspuru-Guzik Group, University of Toronto.

import os, sys
import glob, copy
import uuid, time

import threading
import multiprocessing
import subprocess

import yaml
import pickle
import numpy as np 

from file_logger import FileLogger


#========================================================

MAX_ITERS        = 1000
BATCH_SIZE       = 4
MAX_JOBS_RUNNING = 1

TEMPLATE_SETTINGS_FILE        = 'template_settings.yml'
TEMPLATE_EXECUTABLE_FILE      = 'MNIST_PyT_hparams.py'

#========================================================

def thread(function):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target = function, args = args, kwargs = kwargs)
        thread.start()
    return wrapper

def process(function):
    def wrapper(*args, **kwargs):
        process = multiprocessing.Process(target = function, args = args, kwargs = kwargs)
        process.start()
    return wrapper

#========================================================



def get_optimization_domain():
    domain = [
            {'name': 'n_chanels_output_layer1', 'type': 'discrete', 'domain': tuple(range(5, 30))},
            {'name': 'n_chanels_output_layer2', 'type': 'discrete', 'domain': tuple(range(5, 50))},
            {'name': 'filtersize_layer1', 'type': 'discrete', 'domain': tuple(range(2, 9))},
            {'name': 'filtersize_layer2', 'type': 'discrete', 'domain': tuple(range(2, 9))},
            {'name': 'number_of_neurons_dense_layer1', 'type': 'discrete', 'domain': tuple(range(5,500))},
            {'name': 'number_of_neurons_dense_layer2', 'type': 'discrete', 'domain': tuple(range(5,500))},
            {'name': 'momentum', 'type': 'continuous', 'domain': (-5, -1)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (-5, -1)},
            {'name': 'batch_size_train', 'type': 'discrete', 'domain': tuple(range(5, 500))}
            ]
    return domain

#========================================================

class Manager(object):

    CACHE_PARAMS = []
    FILE_LOGGERS = {}

    def __init__(self, 
            max_iters              = MAX_ITERS, 
            batch_size             = BATCH_SIZE, 
            max_jobs_running       = MAX_JOBS_RUNNING,
            template_settings_file = TEMPLATE_SETTINGS_FILE,
            template_executable_file      = TEMPLATE_EXECUTABLE_FILE,
        ):

        self.num_iters        = 0
        #self.cuda_device      = 0
        self.max_iters        = max_iters
        self.batch_size       = batch_size
        self.max_jobs_running = max_jobs_running 
        self.hyperopt_domain  = get_optimization_domain()

        self.num_iter             = 0
        self.num_jobs_running     = 0
        self.updating_param_cache = False
        self.is_training          = False
        self.has_new_feedback     = False

        with open(template_settings_file, 'r') as content:
            self.template_settings = yaml.load(content)
        with open(template_executable_file, 'r') as content:
            self.template_executable = content.read()

        self.submitted_params = {}
        self.all_params = []
        self.all_maes = []
        #self.all_divers = []
        #self.all_recons = []


        #self.chimera    = Chimera(tolerances)


    def get_bo_instance(self, evaluator = lambda x: x, params = None, measurements = None):
        from GPyOpt.methods import BayesianOptimization
        bo = BayesianOptimization(
                f                = evaluator,
                domain           = self.hyperopt_domain,
                batch_size       = self.batch_size,
                exact_feval      = True,
                model_type       = 'GP_MCMC', 
                acquisition_type = 'EI_MCMC',
                X = params, Y = measurements,
            )    
        return bo


    #--- PARAM MANAGEMENT ---#

    def get_params(self):
        while len(self.CACHE_PARAMS) == 0:
            if not self.is_training: 
#                self. ... ### GENERATE NEW PARAMETERS ###
                print("ERROR! NO PARAMS IN CACHE AND NO TRAINING IS BEING DONE!")
                pass
            time.sleep(0.1)    
        while self.updating_param_cache:
            time.sleep(0.1)
        self.updating_param_cache = True
        params = self.CACHE_PARAMS.pop(0)
        self.updating_param_cache = False
        return params

    def update_param_cache(self, params):
        self.updating_param_cache = True
        self.CACHE_PARAMS         = [param for param in params]
        self.updating_param_cache = False

    #--- JOB MANAGEMENT ---#

    def _create_new_job(self):
        job_id = str(uuid.uuid4())[:8]
        params = self.get_params()
        param_dict = {}
        param_dict = {}
        for entry_index, entry in enumerate(self.hyperopt_domain):
            param_dict[entry['name']] = params[entry_index].item()
        param_dict['momentum'] = 10**param_dict['momentum']
        param_dict['learning_rate'] = 10**param_dict['learning_rate']

        # write new settings file 
        settings = copy.deepcopy(self.template_settings)
        for prop in ['n_chanels_output_layer1','n_chanels_output_layer2','filtersize_layer1',
                     'filtersize_layer2','number_of_neurons_dense_layer1','number_of_neurons_dense_layer2']:
            settings['model'][prop] = int(param_dict[prop])

        settings['training']['momentum'] = param_dict['momentum']
        settings['training']['learning_rate'] = param_dict['learning_rate']
        settings['training']['batch_size_train'] = param_dict['batch_size_train']
        #cuda_device                                 = self.cuda_device % self.max_jobs_running
        #self.cuda_device                        += 1 
        #settings['data']['cuda_device']          = cuda_device

        # create new environment
        dir_name = '/home/xieruo1/Documents/hparam_optimization/Submissions/Simulation_%s' % job_id
        os.mkdir(dir_name)
        with open('%s/template_settings.yml' % dir_name, 'w') as content:
            yaml.dump(settings, content, default_flow_style = False)
        with open('%s/MNIST_PyT_hparams.py' % dir_name, 'w') as content:
            content.write(self.template_executable)

        self.log_submission(job_id, param_dict, params)
        return job_id, dir_name


    def log_submission(self, job_id, parameters, param_array):
        file_name = 'job_submissions.pkl'
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as content:
                submitted = pickle.load(content)
        else:
            submitted = []
        sub = {'job_id': job_id, 'params': parameters}
        submitted.append(sub)
        with open(file_name, 'wb') as content:
            pickle.dump(submitted, content)
        self.submitted_params[job_id] = param_array


    def _submit_job(self):
        self.num_jobs_running += 1
        job_id, dir_name = self._create_new_job()

        # prepare file logger
        file_logger = FileLogger(action = self.parse_results, path = dir_name, pattern = '*COMPLETED*')
        file_logger.start()
        self.FILE_LOGGERS[job_id] = file_logger
        
        # submit job
#        subprocess.call('python %s/vae_rnn.py %s' % (dir_name, dir_name), shell = True)
        import shlex
        print("START THE JOB")
        command = 'python %s/MNIST_PyT_hparams.py %s' % (dir_name, dir_name)
        args    = shlex.split(command)
#        process = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        process = subprocess.Popen(args)


    #--- MEASUREMENTS MANAGEMENT ---#

    def parse_results(self, file_name):
        dir_name = '/'.join(file_name.split('/')[:-1])
        job_id   = dir_name.split('_')[-1]

        # stop filelogger
        self.FILE_LOGGERS[job_id].stop()
        del self.FILE_LOGGERS[job_id]

        print('FOUND FILE NAME', file_name)

        # parse results
        #valids, divers, recons = [], [], []
        maes = []
        results_file  = '%s/results.dat' % dir_name
        with open(results_file, 'r') as content:
            for line in content:
                #linecontent = line.split('|')
                #valids.append(float(linecontent[0].split()[1]))
                #divers.append(float(linecontent[1].split()[1]))
                #recons.append(float(linecontent[2].split()[1]))
                maes.append(float(line.split()[0]))
        #valids = np.array(valids)
        #divers = np.array(divers)
        #recons = np.array(recons)
        maes = np.array(maes)
        #valids = np.where(np.isnan(valids), 0., valids)
        #divers = np.where(np.isnan(divers), 0., divers)
        #recons = np.where(np.isnan(recons), 0., recons)
        maes = np.where(np.isnan(maes), 0., maes)
        best_index = np.argmin(maes)
        #valid, diver, recon = valids[best_index], divers[best_index], recons[best_index]
        mae = maes[best_index]
    
        print('FOUND', mae)

        self.all_params.append(self.submitted_params[job_id])
        #self.all_valids.append(valid)
        #self.all_divers.append(diver)
        #self.all_recons.append(recon)
        self.all_maes.append(mae)

        self.has_new_feedback  = True
        self.num_jobs_running -= 1



    #--- BO MANAGEMENT ---#        

    def sample_first_params(self):
        def evaluator(params):
            return 0.
        bo     = self.get_bo_instance(evaluator = evaluator)
        params = bo.get_evaluations()[0]
        print("first evaluation:")
        print(params)
        self.update_param_cache(params)

    
    @thread
    def sample_params(self):
        self.is_training      = True
        self.has_new_feedback = False

        # collect measurements
        params = np.array(self.all_params)


        #self.scalarized = 0.5 * np.array(self.all_valids) + 0.3 * np.array(self.all_recons) + 0.2 * np.array(self.all_divers)
        #objs   = [self.all_valids, self.all_recons, self.all_divers, self.scalarized]
        #objs   = - np.array(objs)

        #merits = self.chimera.scalarize(objs.T) 
        #merits = (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))
        #merits = np.reshape(merits, (len(merits), 1))

        objs   = self.all_maes
        objs   = np.array(objs)
        print("objs.shape: ", objs.shape)
        print(objs)
        merits = objs
        merits = (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))
        merits = np.reshape(merits, (len(merits), 1))

        print("params.shape: ", params.shape)
        print(params)
        print("merits.shape: ", merits.shape)
        print(merits)


        with open('intermediate_results.pkl', 'wb') as content:
            pickle.dump({'params': params, 'measurements': objs.T, 'merits': merits}, content)

        bo = self.get_bo_instance(params = params, measurements = merits)
        try:
            params = bo.suggest_next_locations()
        except np.linalg.LinAlgError:
            pass
        self.update_param_cache(params)
        self.is_training = False


    #--- CLOSED LOOP CONTROL ---#

    def run_hyperopt(self):
        self.sample_first_params()
        
        while self.num_iter < self.max_iters:

            # check if we can submit new jobs
            if self.num_jobs_running < self.max_jobs_running:
                self._submit_job()
                self.num_iters += 1

            # check if we received some feedback
            print(self.has_new_feedback, self.is_training)
            if self.has_new_feedback and not self.is_training:
#                self.has_new_feedback = False
                self.sample_params()

            import time
            time.sleep(1)

#========================================================


if __name__ == '__main__':

    manager = Manager()
    manager.run_hyperopt()

