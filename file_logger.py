#!/usr/bin/env python 
# Written by Dr. Pascal Frederich and later modified.
# From the Aspuru-Guzik Group, University of Toronto.


__author__ = 'Florian Hase'

#==============================================================

import pickle

from watchdog.observers import Observer
from watchdog.events    import PatternMatchingEventHandler

#==============================================================

class FileHandler(PatternMatchingEventHandler):
	
	def __init__(self, event, pattern):
		PatternMatchingEventHandler.__init__(self, patterns = [pattern])
		self.process_event = event
	
	def process(self, found_file):
		file_name = found_file.src_path
		self.process_event(file_name)

	def on_created(self, found_file):
		self.process(found_file)


class FileModifiedHandler(PatternMatchingEventHandler):
	
	def __init__(self, event, pattern):
		PatternMatchingEventHandler.__init__(self, patterns = [pattern])
		self.process_event = event

	def process(self, found_file):
		file_name = found_file.src_path
		self.process_event(file_name)

	def on_modified(self, found_file):
		self.process(found_file)

#==============================================================

class FileLogger(object):

	def __init__(self, action, path = './', pattern = '*'):
		self.path          = path
		self.event_handler = FileHandler(action, pattern)

	def start(self):
		self.observer = Observer()
		self.observer.schedule(self.event_handler, self.path, recursive = True)
		self.observer.start()

	def stop(self):
		self.observer.stop()



class FileObserver(object):
	
	def __init__(self, action, path = './', pattern = '*'):
		self.path          = path
		self.event_handler = FileModifiedHandler(action, pattern)

	def start(self):
		self.observer = Observer()
		self.observer.schedule(self.event_handler, self.path, recursive = True)
		self.observer.start()

	def stop(self):
		self.observer.stop() 

#==============================================================

