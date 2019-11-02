
import tensorflow as tf
from tensorflow_train.utils.data_format import get_image_axes, get_channel_index
import numpy as np

def softmax(labels, logits, weights=None,  data_format='channels_first'):
    channel_index = get_channel_index(labels, data_format)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=channel_index)
    return tf.reduce_mean(loss)


def spread_loss(labels,logits,m_low=0.2,m_hight=0.9,iteration_low_to_high=100000,global_step=100000,data_format='channels_first'):
    m=m_low+(m_hight-m_low)*tf.minimum(tf.to_float(global_step/iteration_low_to_high),tf.to_float(1))
    n_labels=labels.get_shape()[1]
    labels=tf.transpose(labels,(1, 0, 2, 3))
    logits=tf.transpose(logits,(1, 0, 2, 3))
    labels=tf.manip.reshape(labels,[n_labels,-1])  
    logits=tf.manip.reshape(logits,[n_labels,-1])

    true_class_logits=tf.reduce_max(labels*logits,axis=0)
    margin_loss_pixel_class=tf.square(tf.nn.relu((m-true_class_logits+logits)*(1-labels)))

    loss=tf.reduce_mean(tf.reduce_sum(margin_loss_pixel_class,axis=0))

    return loss


def weighted_spread_loss(labels,logits,m_low=0.2,m_hight=0.9,iteration_low_to_high=100000,global_step=100000, data_format='channels_first',
                         w_l=np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978]) * 5):
  
  channel_index = get_channel_index(labels, data_format)
  
  m=m_low+(m_hight-m_low)*tf.minimum(tf.to_float(global_step/iteration_low_to_high),tf.to_float(1))
  n_labels=labels.get_shape()[1]
  labels=tf.transpose(labels,[1, 0, 2, 3])
  logits=tf.transpose(logits,[1, 0, 2, 3])
  labels=tf.manip.reshape(labels,[n_labels,-1])  
  logits=tf.manip.reshape(logits,[n_labels,-1])  

  true_class_logits=tf.reduce_max(labels*logits,axis=0)
  margin_loss_pixel_class=tf.square(tf.nn.relu((m-true_class_logits+logits)*tf.abs(labels-1)))

  loss=[]
  for i in range(len(w_l)):
    loss.append(w_l[i]*margin_loss_pixel_class*tf.gather(labels,[i],axis= channel_index))

  loss=tf.reduce_sum(tf.stack(loss,axis=0),axis=0)
  loss=tf.reduce_mean(tf.reduce_sum(loss,axis=0))

  return loss

def weighted_softmax(labels, logits, data_format='channels_first',
                     w_l=np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978]) * 5):
  
  channel_index = get_channel_index(labels, data_format)
  loss_s=tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits, dim = channel_index)

  loss=[]
  for i in range(len(w_l)):
    loss.append(w_l[i] * loss_s * tf.gather(labels,[i],axis = channel_index))

  loss=tf.reduce_sum(tf.stack(loss,axis=0),axis=0)
  return tf.reduce_mean(loss)

def focal_loss_fixed(y_true,logits,gamma=2., alpha=4.,data_format='channels_first'):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
        
    NB for this class the funciton will be of the shape batch size, num cls, height,width as these are images
        
    """
    
    channel_index = get_channel_index(y_true, data_format)
    
    y_pred=tf.nn.softmax(logits,axis=channel_index)
    
    gamma = float(gamma)
    alpha = float(alpha)
    
    epsilon = 1.e-9
    #y_true = tf.convert_to_tensor(y_true, tf.float32)
    #y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    
    reduced_fl = tf.reduce_max(fl, axis=1)
    
    return tf.reduce_mean(reduced_fl)
