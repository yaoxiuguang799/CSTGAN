import os
import h5py
import numpy as np
from data_loader import CSTGANDataLoader
from model.CSTGAN.CSTGAN import CSTGAN


if __name__ == '__main__':
    seq_len = 7
    img_size = 64
    rnn_hidden_size = 256
    batch_size = 64
    epochs = 100
    train_path = './data/dataset/Train.h5'
    val_path = './data/dataset/Val.h5'
    test_dir = './data/dataset'
    out_dir = './data/predict'
    
    is_Train = False
    saving_path = f'./geneModel/CSTGAN'
    trained_model_path = f'./geneModel/CSTGAN/CSTGAN_best.pt'
    
    if is_Train:
        os.makedirs(saving_path, exist_ok=True)
        # -------- Create model --------
        model = CSTGAN(seq_len=seq_len,img_size=img_size,rnn_hidden_size=rnn_hidden_size,epochs=epochs,saving_path=saving_path)
        # -------- Train step --------
        dataset = CSTGANDataLoader(train_dataset_path=train_path,val_dataset_path=val_path,test_dataset_path=None,batch_size=batch_size)
        model.train(dataset=dataset)
    else:
        # -------- Create model --------
        model = CSTGAN(seq_len=seq_len,img_size=img_size,rnn_hidden_size=rnn_hidden_size,epochs=epochs,trained_model_path=trained_model_path,saving_path=saving_path)
        
    # -------- Test step --------
    acc_metrics = []
    for r in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]:
        test_path = os.path.join(test_dir, f"Test_MissingRatio_{r}.h5")
        dataset = CSTGANDataLoader(train_dataset_path=None,val_dataset_path=None,test_dataset_path=test_path,batch_size=batch_size)
        imputation_data, test_r, test_bias, test_mae, test_rmse, test_mre = model.predict(dataset=dataset)
        acc_metrics.append([float(r),test_r, test_bias, test_mae, test_rmse, test_mre])
        print(
            f"test Missing Ratio: {r}, "
            f"test R: {test_r:.4f}, "
            f"test bias: {test_bias:.4f}mm, "
            f"test MAE: {test_mae:.4f}mm, "
            f"test RMSE: {test_rmse:.4f}mm, "
            f"test MRE: {test_mre:.4f}%"
        )

        imputation_path = os.path.join(out_dir, f"Test_CSTGAN_imputation_{r}.h5")
        with h5py.File(imputation_path, "w") as hf:
            hf.create_dataset("X", data=imputation_data, compression="gzip", compression_opts=4)
    acc_metrics = np.array(acc_metrics)
    metrics_path = os.path.join(out_dir, f"test_CSTGAN_metrics.csv")
    fmt = ["%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f"]
    np.savetxt(metrics_path,acc_metrics,delimiter=",",header="ClearRatio,R,Bias,MAE,RMSE,MRE",fmt=fmt)
    print('finish!!!')
