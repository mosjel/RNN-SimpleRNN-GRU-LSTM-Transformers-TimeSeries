import tensorflow as tf
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
s=tf.range(start=0,limit=10,delta=1)
print(s)