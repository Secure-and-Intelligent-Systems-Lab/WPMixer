
from tabulate import tabulate
import csv
import datetime
from collections import defaultdict

class Output_database:
    def __init__(self):
        self.data = {
                'data': [],
                'seq_len': [],
                'pred_len': [],
                'mse': [],
                'std_mse': [],
                'mae': [],
                'std_mae': [],
                'min_mse': [],
                'std_min_mse': [],
                'min_mae': [],
                'std_min_mae': [],
                'mpiw': [],
                'std_mpiw': [],
                'picp': [],
                'std_picp': []
                }
        self.data = defaultdict(list)
        self.current_time = datetime.datetime.now()
        self.current_time = str(self.current_time).replace(':', '-').split('.')[0]

    def push(self, data0, seq_len0, pred_len0, mse0, std_mse0, mae0, std_mae0, minmse0, std_minmse0, minmae0, std_minmae0, mpiw0, std_mpiw0, picp0, std_picp0):
        self.data['data'].append(data0)
        self.data['seq_len'].append(seq_len0)
        self.data['pred_len'].append(pred_len0)
        
        self.data['mse'].append(mse0)
        self.data['std_mse'].append(std_mse0)
        
        self.data['mae'].append(mae0)
        self.data['std_mae'].append(std_mae0)
        
        self.data['min_mse'].append(minmse0)
        self.data['std_min_mse'].append(std_minmse0)
        
        self.data['min_mae'].append(minmae0)
        self.data['std_min_mae'].append(std_minmae0)
        
        self.data['mpiw'].append(mpiw0)
        self.data['std_mpiw'].append(std_mpiw0)
        
        self.data['picp'].append(picp0)
        self.data['std_picp'].append(std_picp0)
        
    def generate_table(self):
        data_list = [dict(zip(self.data.keys(), [self.data[key][i] for key in self.data.keys()])) for i in range(len(self.data['data']))]
        print(tabulate(data_list, headers="keys", tablefmt="pretty"))
        
    def save(self, args_params = None):
        if args_params is not None:
            file_name = '{}_decomposition-{}'.format(args_params.model, not args_params.no_decomposition)
        else:
            file_name = 'Model'
        
        csv_file = './outputs/' + file_name + '_' + self.current_time + '.csv' # sys.path[-1] + '/database_output.csv'
        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            # Create a CSV writer object
            csv_writer = csv.writer(file)
            # Write the header (keys of the dictionary) to the CSV file
            csv_writer.writerow(self.data.keys())
            # Write the values of the dictionary to the CSV file
            for row in zip(*self.data.values()):
                csv_writer.writerow(row)
        print(f"The dictionary has been successfully saved to {csv_file}")
        
class Output_database2:
    def __init__(self):
        self.data = {
                'data': [],
                'seq_len': [],
                'pred_len': [],
                'mse_mean': [],
                'mse_std': [],
                'mae_mean': [],
                'mae_std': [],
                'min_mse_mean': [],
                'min_mse_std': [],
                'min_mae_mean': [],
                'min_mae_std': []
                }
        self.decimal = 3

    def push(self, data0, seq_len0, pred_len0, mse0_mean, mse0_std, mae0_mean, mae0_std, minmse0_mean, minmse0_std, minmae0_mean, minmae0_std):
        self.data['data'].append(data0)
        self.data['seq_len'].append(seq_len0)
        self.data['pred_len'].append(pred_len0)
        self.data['mse_mean'].append(round(mse0_mean, self.decimal))
        self.data['mse_std'].append(round(mse0_std, self.decimal))
        self.data['mae_mean'].append(round(mae0_mean, self.decimal))
        self.data['mae_std'].append(round(mae0_std, self.decimal))
        self.data['min_mse_mean'].append(round(minmse0_mean, self.decimal))
        self.data['min_mse_std'].append(round(minmse0_std, self.decimal))
        self.data['min_mae_mean'].append(round(minmae0_mean, self.decimal))
        self.data['min_mae_std'].append(round(minmae0_std, self.decimal))
        
    def generate_table(self):
        data_list = [dict(zip(self.data.keys(), [self.data[key][i] for key in self.data.keys()])) for i in range(len(self.data['data']))]
        print(tabulate(data_list, headers="keys", tablefmt="pretty"))
        
    def save(self):
        current_time = datetime.datetime.now()
        current_time = str(current_time).replace(':', '-').split('.')[0]
        csv_file = './database_output.csv' # sys.path[-1] + '/database_output.csv'
        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            # Create a CSV writer object
            csv_writer = csv.writer(file)
            # Write the header (keys of the dictionary) to the CSV file
            csv_writer.writerow(self.data.keys())
            # Write the values of the dictionary to the CSV file
            for row in zip(*self.data.values()):
                csv_writer.writerow(row)
        print(f"The dictionary has been successfully saved to {csv_file}")