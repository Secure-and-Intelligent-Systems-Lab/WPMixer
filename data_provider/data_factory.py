from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_M4, Dataset_Solar
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1':Dataset_ETT_hour,
    'ETTh2':Dataset_ETT_hour,
    'ETTm1':Dataset_ETT_minute,
    'ETTm2':Dataset_ETT_minute,
    'Weather':Dataset_Custom,
    'Traffic':Dataset_Custom,
    'Electricity':Dataset_Custom,
    'ILI':Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed!='timeF' else 1

#     if flag == 'test':
#         shuffle_flag = False; 
#         drop_last = True;
#         batch_size = args.batch_size;
#         freq=args.freq
#     elif flag=='pred':
#         shuffle_flag = False; 
#         drop_last = False;
#         batch_size = 1; 
#         freq=args.detail_freq
#         Data = Dataset_Pred
#     else:
#         shuffle_flag = True;
#         drop_last = True; 
#         batch_size = args.batch_size; 
#         freq=args.freq
    
#     data_set = Data(root_path=args.root_path,
#                     data_path=args.data_path,
#                     flag=flag,
#                     size=[args.seq_len, None, args.pred_len], # [args.seq_len, args.label_len, args.pred_len]
#                     features=args.features,
#                     target=args.target,
#                     inverse=None, # args.inverse
#                     timeenc=timeenc,
#                     freq=freq,
#                     cols=args.cols)
    
#     data_loader = DataLoader(data_set,
#                              batch_size=batch_size,
#                              shuffle=shuffle_flag,
#                              num_workers=args.num_workers,
#                              drop_last=drop_last)
        

#     print(flag, len(data_set))
#     return data_set, data_loader
