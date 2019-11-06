
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
                         w_l=np.array([0.001259780,0.03987201, 0.36867433, 0.35872208, 0.2314718])):
  
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
                     w_l=np.array([0.00125978,0.03987201, 0.36867433, 0.35872208, 0.2314718])):
  
  channel_index = get_channel_index(labels, data_format)
  loss_s=tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits, dim = channel_index)

  loss=[]
  for i in range(len(w_l)):
    loss.append(w_l[i] * loss_s * tf.gather(labels,[i],axis = channel_index))

  loss=tf.reduce_sum(tf.stack(loss,axis=0),axis=0)
  return tf.reduce_mean(loss)

def focal_loss_fixed(labels,logits,gamma=2., alpha=4.,data_format='channels_first'):
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
    y_true=labels
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

def generalised_dice_loss(logits,
                          labels,
                          weight_map=None,
                          type_weight='Square',
                          data_format='channels_first'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    
    channel_index = get_channel_index(labels, data_format)
    prediction = tf.nn.softmax(logits,axis=channel_index)
    one_hot=labels
    
    #ipdb.set_trace()
    #if len(ground_truth.shape) == len(prediction.shape):
     #   ground_truth = ground_truth[..., -1]
    #one_hot =tf.cast(ground_truth,tf.float32)
    #prediction=tf.cast(prediction,tf.float32)

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.reduce_sum(one_hot, 0)
        intersect = tf.reduce_sum(one_hot * prediction,0)
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score
