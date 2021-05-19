import os
import time
import tensorflow as tf
from models import build_model
from core import build_metric
from data import build_dataset
from core import build_optimizer
from core import LookaheadOptimizer
from core import build_learning_rate_scheduler


def _time_to_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


class MultiGPUTrainer(object):
    """Train class.

        Args:
            detector: detector.
            strategy: Distribution strategy in use.
            cfg: the configuration cfg.
        """

    def __init__(self, cfg):
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_replicas = self.strategy.num_replicas_in_sync
        cfg.train.dataset.batch_size *= self.num_replicas 
        cfg.val.dataset.batch_size *= self.num_replicas
        train_dataset = build_dataset(**cfg.train.dataset.as_dict())
        val_dataset = build_dataset(**cfg.val.dataset.as_dict())

        with self.strategy.scope():
            self.train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
            self.val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

            use_mixed_precision = cfg.dtype == "float16"
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)

            self.model = build_model(**cfg.model.as_dict())

            optimizer = build_optimizer(**cfg.optimizer.as_dict())
        
            if cfg.lookahead:
                optimizer = LookaheadOptimizer(optimizer, cfg.lookahead.steps, cfg.lookahead.alpha) 

            if use_mixed_precision:
                optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer=optimizer, loss_scale= "dynamic") 
            
            self.loss_fn = build_loss(**cfg.loss.as_dict())

            self.optimizer = optimizer
            self.use_mixed_precision = use_mixed_precision
            self.cfg = cfg

            self.total_train_steps = cfg.learning_rate_scheduler.train_steps
            self.learning_rate_scheduler = build_learning_rate_scheduler(
                **cfg.learning_rate_scheduler.as_dict())

            self.global_step = tf.Variable(initial_value=0,
                                        trainable=False,
                                        name="global_step",
                                        dtype=tf.int64)

            self.learning_rate = tf.Variable(initial_value=0,
                                            trainable=False,
                                            name="learning_rate",
                                            dtype=tf.float32)
        
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                    directory=cfg.checkpoint_dir,
                                                    max_to_keep=10)
            if os.path.exists(cfg.pretrained_weights_path): 
                self.model.load_weights(cfg.pretrained_weights_path, by_name=True, skip_mismatch=True)

            latest_checkpoint = self.manager.latest_checkpoint
            if latest_checkpoint is not None:
                try:
                    steps = int(latest_checkpoint.split("-")[-1])
                    self.global_step.assign(steps)
                except:
                    self.global_step.assign(0)
                self.checkpoint.restore(latest_checkpoint)
                tf.print(_time_to_string(), "Restored weights from %s." % latest_checkpoint)
            else:
                self.global_step.assign(0)

            self.summary_writer = tf.summary.create_file_writer(logdir=cfg.summary_dir)
            self.log_every_n_steps = cfg.log_every_n_steps
            self.save_ckpt_steps = cfg.save_ckpt_steps
            self.use_jit = tf.config.optimizer.get_jit() is not None

            self.training_loss_metrics = {}
            self.val_loss_metrics = {}

            self.train_acc_metric = tf.keras.metrics.Accuracy() 
            self.train_auc_metric = tf.keras.metrics.AUC()
            self.val_acc_metric = tf.keras.metrics.Accuracy() 
            self.val_auc_metric = tf.keras.metrics.AUC()
            self._add_graph = True

    def run(self):
        with self.strategy.scope():
            def compute_loss(logits, labels):
                with tf.name_scope("compute_loss"):
                    logits = tf.cast(logits, tf.float32)
                    if self.cfg.num_classes == 1:
                        labels = tf.cast(tf.expand_dims(labels, -1), tf.float32)
                    else:
                        labels = tf.one_hot(labels, self.cfg.num_classes)
                    ce = self.loss_fn(labels, logits)
                    ce = tf.reduce_mean(ce)

                    l2_loss = tf.add_n(
                        [tf.nn.l2_loss(w) for w in self.model.trainable_weights if "kernel" in w.name])
                    l2_loss *= self.cfg.weight_decay

                    return dict(cross_entropy=ce, l2_loss=l2_loss)

            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_train_step(images, labels):
                # @tf.function(experimental_relax_shapes=True)
                def step_fn(batch_images, batch_labels):
                    with tf.name_scope("train_step"):
                        with tf.GradientTape(persistent=True) as tape:
                            logits = self.model(batch_images, training=True)

                            loss_dict = compute_loss(logits, batch_labels)

                            loss = tf.add_n([v for k, v in loss_dict.items()])
                            loss_dict["loss"] = loss
                            if self.use_mixed_precision:
                                scaled_loss = self.optimizer.get_scaled_loss(loss)
                            else:
                                scaled_loss = loss
                        
                        self.optimizer.learning_rate = self.learning_rate.value()
                        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                        if self.use_mixed_precision:
                            gradients = self.optimizer.get_unscaled_gradients(gradients)
                        
                        if self.cfg.gradient_clip_norm > 0.0:
                            gradients, _ = tf.clip_by_global_norm(gradients, self.cfg.gradient_clip_norm)
                        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
                        for key, value in loss_dict.items():
                            if key not in self.training_loss_metrics:
                                self.training_loss_metrics[key] = tf.keras.metrics.Mean()
                            self.training_loss_metrics[key].update_state(value)
                        
                        logits = tf.cast(logits, tf.float32)
                        if self.cfg.num_classes == 1 or self.cfg.use_sigmoid:
                            pred = tf.nn.sigmoid(logits)
                            self.train_auc_metric.update_state(
                                y_pred=pred, y_true=tf.cast(tf.expand_dims(batch_labels, -1), tf.float32))
                            self.train_acc_metric.update_state(
                                y_pred=tf.cast(pred > 0.5), y_true=batch_labels)
                        else:
                            pred = tf.argmax(logits, -1)
                            self.train_acc_metric.update_state(batch_labels, pred)
                            
                        return loss
                
                per_replica_losses = self.strategy.run(step_fn, args=(images, labels))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_losses,
                                            axis=None)

            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_valuate_step(images, labels):
                def step_fn(batch_images, batch_labels):
                    with tf.name_scope("val_step"):
                        logits = self.model(batch_images, training=True)
                        loss_dict = compute_loss(logits, batch_labels)
                        loss = tf.add_n([v for k, v in loss_dict.items()])
                        loss_dict["loss"] = loss

                        for key, value in loss_dict.items():
                            if key not in self.val_loss_metrics:
                                self.val_loss_metrics[key] = tf.keras.metrics.Mean()
                            
                            self.val_loss_metrics[key].update_state(value)
                        
                        logits = tf.cast(logits, tf.float32)
                        if self.cfg.num_classes == 1 or self.cfg.use_sigmoid:
                            pred = tf.nn.sigmoid(logits)
                            self.val_auc_metric.update_state(
                                y_pred=pred, y_true=tf.cast(tf.expand_dims(batch_labels, -1), tf.float32))
                        else:
                            pred = tf.argmax(logits, -1)
                            self.val_acc_metric.update_state(batch_labels, pred)

                return self.strategy.run(step_fn, args=(images, labels))
            

            count = 0
            # TRAIN LOOP
            start = time.time()
            for images, image_info in self.train_dataset:
                self.global_step.assign_add(1)
                lr = self.learning_rate_scheduler(self.global_step.value())
                self.learning_rate.assign(lr)
                if self._add_graph:
                    tf.summary.trace_on(graph=True, profiler=True)
                    distributed_train_step(images, image_info)

                    with self.summary_writer.as_default():
                        tf.summary.trace_export(name=self.cfg.detector,
                                                step=0, profiler_outdir=self.cfg.summary_dir)
                    self._add_graph = False
                else:
                    distributed_train_step(images, image_info)

                count += 1

                info = [_time_to_string(), "TRAINING", self.global_step]
                if tf.equal(self.global_step % self.log_every_n_steps, 0):
                    with self.summary_writer.as_default():
                        for key in self.training_loss_metrics:
                            value = self.training_loss_metrics[key].result()
                            self.training_loss_metrics[key].reset_states()
                            tf.summary.scalar("train/" + key, value, self.global_step)
                            info.extend([key, "=", value])

                        if self.cfg.num_classes == 1 or self.cfg.use_sigmoid:
                            auc = self.train_auc_metric.result()
                            info.extend(["auc =", auc])
                            tf.summary.scalar("train/auc", auc, self.global_step)
                            self.train_auc_metric.reset_states()
                        
                        acc = self.train_acc_metric.result()
                        info.extend(["acc = ", acc])
                        tf.summary.scalar("train/acc", acc, self.global_step)
                        self.train_acc_metric.reset_states()

                        tf.summary.scalar("learning_rate", self.learning_rate.value(), self.global_step)
                        info.extend(["lr", "=", self.learning_rate.value()])
                        
                        tf.summary.image("train/images", images, self.global_step.value(), 5)
                    info.append("(%.2fs)" % ((time.time() - start) / count))
                    tf.print(*info)
                    start = time.time()
                    count = 0

                if self.global_step >= self.total_train_steps:
                    tf.print(_time_to_string(), "Train over.")
                    break

                if self.global_step % self.cfg.val_every_n_steps == 0:
                    tf.print("=" * 150)
                    # EVALUATING LOO
                    val_start = time.time()
                    for images, image_info in self.val_dataset:
                        distributed_valuate_step(images, image_info)

                    val_end = time.time()
                    template = [_time_to_string(), "EVALUATING", self.global_step]
                    with self.summary_writer.as_default():
                        for name in self.val_loss_metrics:
                            value = self.val_loss_metrics[name].result()
                            tf.summary.scalar("val/" + name, value, self.global_step)
                            template.extend([name, "=", value])
                            self.val_loss_metrics[name].reset_states()
                        
                        if self.cfg.num_classes == 1 or self.cfg.use_sigmoid:
                            auc = self.val_auc_metric.result()
                            info.extend(["auc =", auc])
                            tf.summary.scalar("val/auc", auc, self.global_step)
                            self.val_auc_metric.reset_states()
                        
                        acc = self.val_acc_metric.result()
                        info.extend(["acc = ", acc, "(%.2fs)" % (val_end - val_start)])
                        tf.summary.scalar("val/acc", acc, self.global_step)
                        self.val_acc_metric.reset_states()
                    tf.print(*template)
                    
                    self.manager.save(self.global_step)
                    tf.print(_time_to_string(), "Saving detector to %s." % self.manager.latest_checkpoint)

        self.manager.save(self.global_step)
        self.summary_writer.close()
