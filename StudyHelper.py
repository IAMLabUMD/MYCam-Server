import time
import os
from pathlib import Path
from datetime import datetime
from shutil import copyfile

class StudyHelper:
	def __init__(self):
		self.models_hist_dir = '/home/jhong12/TOR-app-files/logs/models_backup'
		
	def GetDirectories(self, dir_path):
		if not os.path.isdir(dir_path):
			return [], []
			
		files = os.listdir(dir_path)
		dnames, dpaths = [], []
		for f in files:
			fpath = os.path.join(dir_path, f)
			if os.path.isdir(fpath):
				dnames.append(f)
				dpaths.append(fpath)
		return dnames, dpaths
		
		
	def GetModelHistory(self, userID):
		user_model_hist_dir = os.path.join(self.models_hist_dir, userID)
		dnames, dpaths = self.GetDirectories(user_model_hist_dir)
	
	def AddModelHistory(self, userID, model_dir, history_type):
		user_model_hist_dir = os.path.join(self.models_hist_dir, userID)
		dnames, dpaths = self.GetDirectories(user_model_hist_dir)
		now = datetime.now()
		dt_string = now.strftime("%Y%m%d_%H_%M_%S")
		save_name = str(len(dnames)+1)+'-'+history_type+'-'+dt_string
		save_dir = os.path.join(self.models_hist_dir, userID, save_name)
		Path(save_dir).mkdir(parents=True, exist_ok=True)  # create the directory
	
		org_model_path = os.path.join(model_dir, 'model.pb')
		org_labels_path = os.path.join(model_dir, 'labels.txt')
		if os.path.isfile(org_model_path):
			copyfile(org_model_path, os.path.join(save_dir, 'model.pb'))
		if os.path.isfile(org_labels_path):
			copyfile(org_labels_path, os.path.join(save_dir, 'labels.txt'))
		return save_name
	
	
if __name__ == '__main__':
	sh = StudyHelper()
	sh.GetModelHistory('D8EE2D49-7117-4675-9736-1C204A8ABA9F')
	sh.AddModelHistory('D8EE2D49-7117-4675-9736-1C204A8ABA9F', '/home/jhong12/TOR-app-files/models/D8EE2D49-7117-4675-9736-1C204A8ABA9F', 'TestCopy')
	sh.GetModelHistory('D8EE2D49-7117-4675-9736-1C204A8ABA9F')