import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  		  	   		   	 		  		  		    	 		 		   		 		  
import timeit
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
  		  	   		   	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it  		  	   		   	 		  		  		    	 		 		   		 		  

def plot_chart(data1, data2, label1, label2, metric, title, file_name):

    leaf_size = np.arange(1, len(data1) + 1)

    plt.figure()
    plt.plot(leaf_size, data1, label=label1)
    plt.plot(leaf_size, data2, label=label2)
    plt.xlabel('Leaf Size')
    plt.ylabel(metric)
    plt.xlim([0, len(data1)])
    plt.legend()
    plt.grid()
    plt.suptitle(title)
    plt.savefig(f'{file_name}.png', bbox_inches='tight')



if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		   	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		   	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		  	   		   	 		  		  		    	 		 		   		 		  
    data = np.genfromtxt(inf, delimiter=',')
    if sys.argv[1] == 'Data/Istanbul.csv':
        data = data[1:, 1:]
  		  	   		   	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		   	 		  		  		    	 		 		   		 		  

    # Add shuffle to break the time series
    np.random.seed(903562473)  # do this only once  		  	   		   	 		  		  		    	 		 		   		 		  
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]


    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  


    # Experiment 1
    rmse_in_sample = np.empty(50, dtype='float')
    rmse_out_sample = np.empty(50, dtype='float')

    for i in range(50):
        learner = dt.DTLearner(leaf_size=i+1)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample  		  	   		   	 		  		  		    	 		 		   		 		  
        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        # evaluate out of sample  		  	   		   	 		  		  		    	 		 		   		 		  
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'DT RMSE between in sample and out of sample', 'RMSE_exp1')


    # Experiment 2
    rmse_in_sample = np.empty(50, dtype='float')
    rmse_out_sample = np.empty(50, dtype='float')

    for i in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':i+1}, bags=1)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'Bagging DT RMSE with bags=1', 'RMSE_bag1_exp2')

    for i in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':i+1}, bags=5)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'Bagging DT RMSE with bags=5', 'RMSE_bag5_exp2')

    for i in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':i+1}, bags=10)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'Bagging DT RMSE with bags=10', 'RMSE_bag10_exp2')

    for i in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':i+1}, bags=20)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'Bagging DT RMSE with bags=20', 'RMSE_bag20_exp2')

    for i in range(50):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':i+1}, bags=50)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse_in_sample[i] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                                                                                            
        pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
        rmse_out_sample[i] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  

    plot_chart(rmse_in_sample, rmse_out_sample, 'In sample', 'Out of sample', 'RMSE',
                'Bagging DT RMSE with bags=50', 'RMSE_bag50_exp2')

    # Experiment 3
    dt_mae = np.empty(50, dtype='float')
    dt_me = np.empty(50, dtype='float')
    dt_time = np.empty(50, dtype='float')

    rt_mae = np.empty(50, dtype='float')
    rt_me = np.empty(50, dtype='float')
    rt_time = np.empty(50, dtype='float')

    for i in range(50):
        start_time = timeit.default_timer()
        dt_learner = dt.DTLearner(leaf_size=i+1)
        dt_learner.add_evidence(train_x, train_y)
        dt_time[i] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        rt_learner = rt.RTLearner(leaf_size=i+1)
        rt_learner.add_evidence(train_x, train_y)
        rt_time[i] = timeit.default_timer() - start_time

        # evaluate out of sample  		  	   		   	 		  		  		    	 		 		   		 		  
        dt_pred_y = dt_learner.query(test_x)
        rt_pred_y = rt_learner.query(test_x)

        dt_mae[i] = abs(test_y - dt_pred_y).mean() 		  	   		   	 		  		  		    	 		 		   		 		  
        rt_mae[i] = abs(test_y - rt_pred_y).mean() 		  	   		   	 		  		  		    	 		 		   		 		  
        dt_me[i] = abs(test_y - dt_pred_y).max() 		  	   		   	 		  		  		    	 		 		   		 		  
        rt_me[i] = abs(test_y - rt_pred_y).max() 		  	   		   	 		  		  		    	 		 		   		 		  
        
    plot_chart(dt_time, rt_time, 'DT', 'RT', 'Time',
                'Training time between RT and DT', 'Time_exp3')
    plot_chart(dt_mae, rt_mae, 'DT', 'RT', 'MAE',
                'MAE between DT and RT', 'MAE_exp3')
    plot_chart(dt_me, rt_me, 'DT', 'RT', 'ME',
                'ME between DT and RT', 'ME_exp3')
