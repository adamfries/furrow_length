import pandas as pd
import sys
import os

condition_dir = sys.argv[1]

root_dir = os.getcwd()

## open up the excel file to write
with pd.ExcelWriter(condition_dir + '.xlsx') as writer:
	## list the folders on the condition directory then seek out 
	##	the csv length files to concatenate into excel sheets
	for i in os.listdir(condition_dir):
		d = os.path.join(root_dir, condition_dir, i)
		if os.path.isdir(d):
			for j in os.listdir(d):
				if j.endswith('lengths_per_time_point.csv'):
					stats = pd.read_csv(os.path.join(d, j))
					stats.to_excel(writer, sheet_name = j)
