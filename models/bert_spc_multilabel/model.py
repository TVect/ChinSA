import tensorflow as tf
import tensorflow_hub as hub
from bert import optimization, modeling


def create_model(is_training, input_ids, input_mask, segment_ids, 
                 label_has_positive, label_has_negative, num_labels, 
                 bert_config):
    """Creates a classification model."""

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    # output_layer = model.get_sequence_output()[:, 0, :]
    output_layer = model.get_pooled_output()

    # control which variables are trainable
    # trainable_variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    # remove_variables = []
    # keep_layers = set(["layer_11", "layer_10", "layer_9", "layer_8", "layer_7", "layer_6"])
    # for variable in trainable_variables:
    #     if all([layer_name not in variable.name for layer_name in keep_layers]):
    #         remove_variables.append(variable)
    # for variable in remove_variables:
    #     trainable_variables.remove(variable)

    def get_intermediate_layer(last_layer, total_layers, desired_layer):
        intermediate_layer_name = last_layer.name.replace(
            str(total_layers + 1), str(desired_layer + 1))
        tf.compat.v1.logging.info(f"Intermediate layer name: {intermediate_layer_name}")
        return tf.get_default_graph().get_tensor_by_name(intermediate_layer_name)

    if is_training:
        output_layer = tf.nn.dropout(output_layer, rate=0.5)

    hidden_size = output_layer.shape[-1].value

    l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)

    def create_att_layer(type_name, input_tensors, masks):
        attw = tf.compat.v1.get_variable(
            f"{type_name}_attw", [hidden_size, 1], 
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            regularizer=l2_regularizer)
        attb = tf.compat.v1.get_variable(
            f"{type_name}_attb", [1], initializer=tf.zeros_initializer())     
        ej = tf.tanh(tf.compat.v1.nn.xw_plus_b(input_tensors, attw, attb))
        # aj = tf.exp(ej) * tf.cast(masks, tf.float32)
        aj = tf.exp(ej) * tf.cast(tf.expand_dims(masks, axis=-1), tf.float32)
        aj = aj / (tf.compat.v1.reduce_sum(aj, axis=1, keepdims=True) + tf.keras.backend.epsilon())
        return aj, tf.reduce_sum(input_tensors * aj, axis=1)

    def create_loss_layer(type_name, input_tensors, class_weights):
        with tf.compat.v1.variable_scope(f"{type_name}_loss"):
            weights = tf.compat.v1.get_variable(
                f"{type_name}_weights", [hidden_size, num_labels],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                regularizer=l2_regularizer)
            bias = tf.compat.v1.get_variable(
                f"{type_name}_bias", [num_labels], 
                initializer=tf.zeros_initializer())

            logits = tf.compat.v1.nn.xw_plus_b(input_tensors, weights, bias)
            probs = tf.nn.softmax(logits, axis=-1, name=f"{type_name}_probs")
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            oh_labels = tf.one_hot(
                label_ids, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(class_weights * oh_labels * log_probs, axis=-1)
            # add regularization loss
            per_example_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, probs


    with tf.compat.v1.variable_scope("output"):
        weights = tf.compat.v1.get_variable(
            "weights", [hidden_size, num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            regularizer=l2_regularizer)
        bias = tf.compat.v1.get_variable(
            "bias", [num_labels], 
            initializer=tf.zeros_initializer())

        logits = tf.compat.v1.nn.xw_plus_b(output_layer, weights, bias)
        probs = tf.nn.sigmoid(logits, name="probs")

        loss_positive = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_has_positive, tf.float32), 
            logits=logits[:, 0], name="loss_positive")
        loss_negative = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_has_negative, tf.float32), 
            logits=logits[:, 1], name="loss_negative")
        total_loss = tf.reduce_mean(loss_positive + loss_negative)

    loss_tensors = {"total": total_loss, 
                    "per_example": loss_positive + loss_negative}
    prob_tensors = {"polarity_pos": probs[:, 0], 
                    "polarity_neg": probs[:, 1]}
    return loss_tensors, prob_tensors


def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps, bert_config, init_checkpoint):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.compat.v1.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_has_positive = features["label_has_positive"]
        label_has_negative = features["label_has_negative"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss_tensors, prob_tensors = create_model(
            is_training, input_ids, input_mask, segment_ids, 
            label_has_positive, label_has_negative, num_labels, bert_config)

        # init from checkpoint
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss_tensors["total"], learning_rate, num_train_steps, num_warmup_steps, False)
            acc_polarity_pos = tf.reduce_mean(
                tf.to_float(
                    tf.equal(label_has_positive , 
                         tf.to_int32(tf.greater(prob_tensors["polarity_pos"], 0.5)))
                ))
            acc_polarity_neg = tf.reduce_mean(
                tf.to_float(
                    tf.equal(label_has_negative , 
                         tf.to_int32(tf.greater(prob_tensors["polarity_neg"], 0.5)))
                ))

            # polarity_pos_preds = tf.argmax(
            #     prob_tensors['polarity_pos'], axis=-1, output_type=tf.int32)
            # polarity_neg_preds = tf.argmax(
            #     prob_tensors['polarity_neg'], axis=-1, output_type=tf.int32)
            # acc_polarity_pos = tf.reduce_mean(
            #     tf.cast(tf.equal(label_has_positive, polarity_pos_preds), tf.float32))
            # acc_polarity_neg = tf.reduce_mean(
            #     tf.cast(tf.equal(label_has_negative, polarity_neg_preds), tf.float32))

            hook_dict = {
                # 'acc_polarity': acc_polarity,
                "acc_polarity_pos": acc_polarity_pos,
                "acc_polarity_neg": acc_polarity_neg,
                'loss': loss_tensors["total"],
                'global_steps': tf.train.get_or_create_global_step()}
            logging_hook = tf.estimator.LoggingTensorHook(hook_dict, every_n_iter=10)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_tensors["total"],
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            pred_positive = tf.cast(tf.greater(prob_tensors["polarity_pos"], 0.5), tf.int32)
            pred_negative = tf.cast(tf.greater(prob_tensors["polarity_neg"], 0.5), tf.int32)
            eval_acc_polarity_pos = tf.compat.v1.metrics.accuracy(
                labels=label_has_positive, 
                predictions=pred_positive)
            eval_acc_polarity_neg = tf.compat.v1.metrics.accuracy(
                labels=label_has_negative, 
                predictions=pred_negative)
            eval_acc_polarity = tf.compat.v1.metrics.accuracy(
                labels=tf.concat([label_has_positive, label_has_negative], axis=-1),
                predictions=tf.concat([pred_positive, pred_negative], axis=-1))

            eval_metric_ops = {
                "eval_acc_polarity_pos": eval_acc_polarity_pos,
                "eval_acc_polarity_neg": eval_acc_polarity_neg,
                "eval_acc_polarity": eval_acc_polarity,
                "eval_loss": tf.compat.v1.metrics.mean(loss_tensors["per_example"])
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_tensors["total"],
                eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions={
                    "polarity_pos_probs": prob_tensors["polarity_pos"], 
                    "polarity_neg_probs": prob_tensors["polarity_neg"]})
        else:
            raise ValueError(
                "Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn
