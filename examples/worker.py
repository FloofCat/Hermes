# These five lines are all you need for the worker script.
import sys
sys.path.append('../hermes/')
import BaselineWorker

worker = BaselineWorker.BaselineWorker()
worker.beginTraining()
