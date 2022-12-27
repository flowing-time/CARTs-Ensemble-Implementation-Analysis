import numpy as np
  		  	   		   	 		  		  		    	 		 		   		 		  
class RTLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  

    def __init__(self, leaf_size=1, verbose=False):  		  	   		   	 		  		  		    	 		 		   		 		  

        self.leaf_size = leaf_size
        self.debug = verbose

  		  	   		   	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "mwu344"  # replace tb34 with your Georgia Tech username  		  	   		   	 		  		  		    	 		 		   		 		  

  		  	   		   	 		  		  		    	 		 		   		 		  
    def build_tree(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, data_y.mean(), -1, -1]], dtype=object)

        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], -1, -1]], dtype=object)

        best_i = np.random.randint(data_x.shape[1])
        split_var = np.median(data_x[:, best_i])

        if split_var == data_x[:, best_i].min() or split_var == data_x[:, best_i].max():
            return np.array([[-1, data_y.mean(), -1, -1]], dtype=object)

        select_left = data_x[:, best_i] <= split_var
        left_tree = self.build_tree(data_x[select_left], data_y[select_left])
        right_tree = self.build_tree(data_x[~select_left], data_y[~select_left])
        root = np.array([[best_i, split_var, 1, left_tree.shape[0] + 1]], dtype=object)

        if self.debug:
            print('root:', root)
            print('left_tree:', left_tree, 'left_tree shape 0: ', left_tree.shape[0])
            print('right_tree', right_tree)
            #print('stack:', np.vstack((root, left_tree, right_tree)))
            pass

        return np.vstack((root, left_tree, right_tree))


    def search_tree(self, X):

        if self.debug:
            print('Search Tree---')
            print('X: ', X)
        
        node = 0
        while True:

            if self.debug:
                print('Node: ', node)
                print(self.dt[node])
                pass

            factor, split_var, left, right = self.dt[node]
                
            if factor == -1:
                return split_var

            elif X[factor] <= split_var:
                node += left
            else:
                node += right

    
    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.dt = self.build_tree(data_x, data_y)

        if self.debug:
            print('Decision Tree size: ', self.dt.shape)
            print(self.dt)
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        #return np.array([ self.search_tree(X) for X in points ])
        
        result = np.empty(points.shape[0], dtype='float')

        for i, X in enumerate(points):
            result[i] = self.search_tree(X)

        return result
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		   	 		  		  		    	 		 		   		 		  
