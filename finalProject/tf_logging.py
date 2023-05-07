import os
from tensorflow.keras.callbacks import TensorBoard
import datetime
class TF_LOGGING:
    def __init__(self):
        pass
    # Create a logfile at directory with logname
    #return a tensorflow callback
    def create(self,directory):
        #check if directory exists
        if(not os.path.exists(directory)):
            os.makedirs(directory)
        #Create a path from directory and create a log file with timestamp            
        logdir = os.path.join(directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return TensorBoard(logdir,histogram_freq=1)