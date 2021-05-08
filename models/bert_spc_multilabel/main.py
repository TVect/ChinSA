import os
import tensorflow as tf
from helper import file_based_input_fn_builder, serving_fn_builder
from model import model_fn_builder
from bert import modeling


flags = tf.flags

FLAGS = flags.FLAGS

FILE_HOME = os.path.abspath(os.path.dirname(__file__))

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "output_dir", os.path.join(FILE_HOME, "./output"),
    "The output directory which contains tfrecords | checkpoints | saved model.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("num_labels", 1, "number of labels")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("keep_checkpoint_max", 3,
                     "The maximum number of recent checkpoint files to keep.")

flags.DEFINE_integer("max_steps_without_decrease", 100,
                     "earlystop: maximum number of training steps with no decrease in the given metric.")

flags.DEFINE_string("warm_start_path", "",
                    "filepath to a checkpoint or SavedModel to warm-start from")

def main(_):
    # tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

    # tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    train_file = os.path.join(FLAGS.output_dir, "tfrecords/train.tf_record")
    eval_file = os.path.join(FLAGS.output_dir, "tfrecords/dev.tf_record")

    train_examples_size = sum(1 for _ in tf.python_io.tf_record_iterator(train_file))

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_steps = int(train_examples_size / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        num_labels=FLAGS.num_labels,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(gpu_options=gpu_options),
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max
    )
    model_params = {"batch_size": FLAGS.batch_size}

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(FLAGS.output_dir, "ckpts"),
        params=model_params,
        config=run_config,
        warm_start_from=FLAGS.warm_start_path if FLAGS.warm_start_path else None
        )


    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=False)
        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=FLAGS.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)
        earlystopping_hook = tf.estimator.experimental.stop_if_no_increase_hook(
            estimator,
            metric_name='eval_acc_polarity',
            max_steps_without_increase=FLAGS.max_steps_without_decrease,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps
        )


        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=num_train_steps,
                                            hooks=[earlystopping_hook])
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
        #                                   steps=None)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # estimator.export_savedmodel(os.path.join(FLAGS.output_dir, "savedmodel"),
        #                             serving_fn_builder(FLAGS.max_seq_length))

        def _acc_bigger(best_eval_result, current_eval_result):
            metric_key = "eval_acc_polarity"
            if not best_eval_result or metric_key not in best_eval_result:
                raise ValueError(
                    'best_eval_result cannot be empty or no metric_key is found in it.')

            if not current_eval_result or metric_key not in current_eval_result:
                raise ValueError(
                    'current_eval_result cannot be empty or no metric_key is found in it.')

            return best_eval_result[metric_key] < current_eval_result[metric_key]


        latest_exporter = tf.estimator.LatestExporter(
            name="models",
            serving_input_receiver_fn=serving_fn_builder(FLAGS.max_seq_length),
            exports_to_keep=3)
        best_exporter = tf.estimator.BestExporter(
            name='best_exporter',
            serving_input_receiver_fn=serving_fn_builder(FLAGS.max_seq_length),
            compare_fn=_acc_bigger,
            exports_to_keep=3)
        exporters = [latest_exporter, best_exporter]
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          throttle_secs=5,
                                          start_delay_secs=5,
                                          steps=None,
                                          exporters=exporters)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)        


    if FLAGS.do_eval:
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # eval_preds = estimator.predict(input_fn=eval_input_fn)

    if FLAGS.do_predict:
        pass


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
