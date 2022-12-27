import LinRegLearner as lrl
import BagLearner as bl
  		  	   		   	 		  		  		    	 		 		   		 		  
class InsaneLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False):  		  	   		   	 		  		  		    	 		 		   		 		  
        self.debug = verbose
        self.learner = bl.BagLearner(learner=bl.BagLearner, bags=20,
                    kwargs={'learner':lrl.LinRegLearner, 'bags':20})
  		  	   		   	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        return "mwu344"
  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        self.learner.add_evidence(data_x, data_y)
  		  	   		   	 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		   	 		  		  		    	 		 		   		 		  
        return self.learner.query(points)