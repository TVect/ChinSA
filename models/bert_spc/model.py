import tensorflow as tf
import tensorflow_hub as hub
from bert import optimization, modeling


def create_model(is_training, input_ids, input_mask, segment_ids, 
                 label_id, num_labels, bert_config):
    """Creates a classification model."""

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    output_layer = model.get_sequence_output()[:, 0, :]

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

        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label_id, tf.float32), 
            logits=logits[:, 0], name="loss")
        loss = tf.reduce_mean(per_example_loss)

    return loss, probs


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
        label_id = features["label_id"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss_tensor, prob_tensor = create_model(
            is_training, input_ids, input_mask, segment_ids, 
            label_id, num_labels, bert_config)

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
                loss_tensor, learning_rate, num_train_steps, num_warmup_steps, False)
            acc = tf.reduce_mean(
                tf.to_float(
                    tf.equal(label_id, 
                         tf.to_int32(tf.greater(prob_tensor[:, 0], 0.5)))
                ))
            hook_dict = {
                'acc': acc,
                # 'prob': prob_tensor,
                # 'label_id': label_id,
                'loss': loss_tensor,
                'global_steps': tf.train.get_or_create_global_step()}
            logging_hook = tf.estimator.LoggingTensorHook(hook_dict, every_n_iter=10)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_tensor,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            pred = tf.to_int32(tf.greater(prob_tensor[:, 0], 0.5))
            eval_acc = tf.compat.v1.metrics.accuracy(
                labels=label_id, 
                predictions=pred)
            eval_metric_ops = {
                "eval_acc": eval_acc,
                "eval_loss": tf.compat.v1.metrics.mean(loss_tensor)
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_tensor,
                eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions={"prob": prob_tensor}
                )
        else:
            raise ValueError(
                "Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn
