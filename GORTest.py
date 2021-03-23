from ObjectRecognizerV2 import ObjectRecognizer
from sklearn.metrics import accuracy_score
import os
from pathlib import Path


pid_list = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 's01', 's02']
# pid_list = ['p01', 'p02', 'p03', 'p04', 'p05']
# pid_list = ['p01', 'p02']
assets_dir = '/home/jhong12/ASSETS2019'
model_dir = '/home/jhong12/GORTestModels/v2'

if not os.path.isdir(model_dir):
	Path(model_dir).mkdir(parents=True, exist_ok=True)

orec = ObjectRecognizer()
orec.debug = True

for pid in pid_list:
	train_data_path = assets_dir + '/' + pid + '/original/train'
	if not os.path.isdir(model_dir+'/'+pid):
		orec.train(model_dir+'/'+pid, train_data_path)
	print(pid, 'training is done.')

orec.debug = False
res = {}
for pid in pid_list:
	test_data_path = assets_dir + '/' + pid + '/original/test1'
	img_dirs = os.listdir(test_data_path)
	
	y_true, y_pred = [], []
	
	for img_dir in img_dirs:
		if os.path.isdir(test_data_path+'/'+img_dir):
			truth = img_dir
			
			for img_file in os.listdir(test_data_path+'/'+img_dir):
				if 'jpg' in img_file:
					best_label, _, _ = orec.predict(model_dir+'/'+pid, test_data_path+'/'+img_dir+'/'+img_file)
					y_true.append(truth)
					y_pred.append(best_label)
					print(pid, truth, best_label)
	
	acc = accuracy_score(y_true, y_pred)
	res[pid] = acc
	print('Accuracy score:', res)


# V1
# Accuracy score: {'p01': 0.38666666666666666, 'p02': 0.37333333333333335, 'p03': 0.5333333333333333, 'p04': 0.48, 'p05': 0.22666666666666666} # epoch 50
# Accuracy score: {'p01': 0.4, 'p02': 0.38666666666666666, 'p03': 0.5066666666666667, 'p04': 0.5866666666666667, 'p05': 0.22666666666666666} # epoch 100
# Accuracy score: {'p01': 0.49333333333333335, 'p02': 0.41333333333333333, 'p03': 0.49333333333333335, 'p04': 0.52, 'p05': 0.2} # epoch 500
# Accuracy score: {'p01': 0.4, 'p02': 0.4266666666666667, 'p03': 0.49333333333333335, 'p04': 0.5333333333333333, 'p05': 0.26666666666666666} # epoch 1000


# V2
# {'p01': 0.49333333333333335, 'p02': 0.08, 'p03': 0.04, 'p04': 0.08, 'p05': 0.0}
# Accuracy score: {'p01': 0.49333333333333335, 'p02': 0.5466666666666666, 'p03': 0.72, 'p04': 0.7333333333333333, 'p05': 0.5333333333333333}
# {'p01': 0.49333333333333335, 'p02': 0.5466666666666666, 'p03': 0.72, 'p04': 0.7333333333333333, 'p05': 0.5333333333333333, 'p06': 0.84, 'p07': 0.9466666666666667, 'p08': 0.5866666666666667, 'p09': 0.7066666666666667, 's01': 0.8533333333333334, 's02': 0.9866666666666667}
