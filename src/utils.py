import datetime
import os

def get_experiment_id():
	return datetime.datetime.now().strftime("%m-%d %H:%M")
