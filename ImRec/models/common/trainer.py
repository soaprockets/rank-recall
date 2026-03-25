# @Time   : 2020/6/26

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']

        # save model
        self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.eval_test_during_training = config['eval_test_during_training']

        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sparseadam':
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        try:
            checkpoint = torch.load(resume_file)
            self.start_epoch = checkpoint['epoch'] + 1
            self.cur_step = checkpoint['cur_step']
            self.best_valid_score = checkpoint['best_valid_score']

            # load architecture params from checkpoint
            if checkpoint['config']['model'].lower() != self.config['model'].lower():
                self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                    'This may yield an exception while state_dict is being loaded.')
            self.model.load_state_dict(checkpoint['state_dict'])

            # load optimizer state from checkpoint only when optimizer type is not changed
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
            self.logger.info(message_output)
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint {resume_file}: {str(e)}")
            raise

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            try:
                # train
                training_start_time = time()
                self.model.pre_epoch_processing()
                train_loss, _ = self._train_epoch(train_data, epoch_idx)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                post_info = self.model.post_epoch_processing()
                if verbose:
                    self.logger.info(train_loss_output)
                    if post_info is not None:
                        self.logger.info(post_info)

                # eval
                if (epoch_idx + 1) % self.eval_step == 0:
                    valid_start_time = time()
                    valid_score, valid_result = self._valid_epoch(valid_data)
                    self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                        valid_score, self.best_valid_score, self.cur_step,
                        max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                    valid_end_time = time()
                    valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                         (epoch_idx, valid_end_time - valid_start_time, valid_score)
                    valid_result_output = 'valid result: \n' + dict2str(valid_result)
                    # test - 只在配置允许时每次验证都评估测试集
                    eval_test_now = self.eval_test_during_training or (stop_flag and test_data is not None)
                    if eval_test_now:
                        test_start_time = time()
                        _, test_result = self._valid_epoch(test_data)
                        test_end_time = time()
                        test_score_output = "epoch %d testing [time: %.2fs]" % \
                                             (epoch_idx, test_end_time - test_start_time)
                        if verbose:
                            self.logger.info(test_score_output)
                            self.logger.info('test result: \n' + dict2str(test_result))
                    else:
                        test_result = None
                    if verbose:
                        self.logger.info(valid_score_output)
                        self.logger.info(valid_result_output)
                    if update_flag:
                        if saved:
                            self._save_checkpoint(epoch_idx)
                            update_output = '██Saving current best: %s' % self.saved_model_file
                            if verbose:
                                self.logger.info(update_output)
                        self.best_valid_result = valid_result

                    if stop_flag:
                        stop_output = 'Finished training, best eval result in epoch %d' % \
                                      (epoch_idx - self.cur_step * self.eval_step)
                        if verbose:
                            self.logger.info(stop_output)
                        break
            except Exception as e:
                self.logger.error(f"Error during training at epoch {epoch_idx}: {str(e)}")
                raise
        return self.best_valid_score, self.best_valid_result


    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            is_test: is in testing?
            idx: current hyper-parameter loop index

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            if model_file:      # load from other file
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            try:
                checkpoint = torch.load(checkpoint_file)
                self.model.load_state_dict(checkpoint['state_dict'])
                message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
                self.logger.info(message_output)
            except Exception as e:
                self.logger.error(f"Failed to load model from {checkpoint_file}: {str(e)}")
                raise

        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -(1 << 10)
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)


    
    
    def _validate_embedding_mapping(self, user_dict, item_dict, user_all_embeddings, item_all_embeddings):
        """
        验证 embedding 和 ID 映射的正确性
        """
        self.logger.info("=" * 60)
        self.logger.info("开始验证 embedding 映射关系...")

        # 1. 检查 embedding 矩阵的 size 是否与 dict 一致
        num_users_in_model = user_all_embeddings.shape[0]
        num_items_in_model = item_all_embeddings.shape[0]
        num_users_in_dict = len(user_dict)
        num_items_in_dict = len(item_dict)

        self.logger.info(f"Model user embeddings: {num_users_in_model}, User dict size: {num_users_in_dict}")
        self.logger.info(f"Model item embeddings: {num_items_in_model}, Item dict size: {num_items_in_dict}")

        assert num_users_in_model == num_users_in_dict, \
            f"User count mismatch: model={num_users_in_model}, dict={num_users_in_dict}"
        assert num_items_in_model == num_items_in_dict, \
            f"Item count mismatch: model={num_items_in_model}, dict={num_items_in_dict}"

        # 2. 检查索引范围
        max_user_idx = max(user_dict.values())
        max_item_idx = max(item_dict.values())

        self.logger.info(f"Max user index in dict: {max_user_idx}, model capacity: {num_users_in_model-1}")
        self.logger.info(f"Max item index in dict: {max_item_idx}, model capacity: {num_items_in_model-1}")

        assert max_user_idx < num_users_in_model, \
            f"User index out of range: max index {max_user_idx} >= model size {num_users_in_model}"
        assert max_item_idx < num_items_in_model, \
            f"Item index out of range: max index {max_item_idx} >= model size {num_items_in_model}"

        # 3. 抽样检查 embedding 合理性
        sample_user_id = list(user_dict.keys())[0]
        sample_user_idx = user_dict[sample_user_id]
        sample_user_emb = user_all_embeddings[sample_user_idx]

        self.logger.info(f"Sample user: original_id={sample_user_id}, internal_idx={sample_user_idx}")
        self.logger.info(f"Sample embedding shape: {sample_user_emb.shape}, "
                         f"mean={sample_user_emb.mean().item():.4f}, "
                         f"std={sample_user_emb.std().item():.4f}")

        # 检查是否有 NaN 或全零
        if torch.isnan(sample_user_emb).any():
            self.logger.warning("WARNING: Sample embedding contains NaN values!")
        if torch.allclose(sample_user_emb, torch.zeros_like(sample_user_emb)):
            self.logger.warning("WARNING: Sample embedding is all zeros!")

        self.logger.info("=" * 60)
        self.logger.info("Embedding mapping validation PASSED!")
        self.logger.info("=" * 60)

    @torch.no_grad()
    def save_all_embedding(self,user_dict,item_dict,load_best_model=True,model_file=None,save_path=None,idx=0):
        """
        Write all user/item embeddings to the disk.

        Args:
        user_dict (dict): A dictionary mapping user IDs or indices to some identifier.
        item_dict (dict): A dictionary mapping item IDs or indices to some identifier.
        load_best_model (bool, optional): Whether to load the best model in the training process. Default is True.
        model_file (str, optional): The saved model file to load. Default is None.
        save_path(str,optional): The path to save the embeddings. Default is None
        idx (int): Current hyper-parameter loop index.

        Returns:
        dict: user/item embedding
        """
        if load_best_model:
            try:
                if model_file:      # load from other file
                    checkpoint_file = model_file
                else:
                    checkpoint_file = self.saved_model_file

                # 检查文件是否存在
                if not os.path.exists(checkpoint_file):
                    self.logger.error(f"Checkpoint file not found: {checkpoint_file}")
                    self.logger.info("Using current model state for embedding extraction")
                else:
                    checkpoint = torch.load(checkpoint_file, map_location=self.device)
                    self.model.load_state_dict(checkpoint['state_dict'])
                    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
                    self.logger.info(message_output)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint for embedding extraction: {str(e)}")
                self.logger.info("Attempting to continue with current model state")

            self.model.eval()

            # Get user and item embeddings
            try:
                # 方法1：尝试使用模型的 get_all_embeddings 方法（适用于图神经网络模型）
                if hasattr(self.model, 'get_all_embeddings'):
                    user_all_embeddings, item_all_embeddings = self.model.get_all_embeddings()
                    self.logger.info("Using model.get_all_embeddings() to extract embeddings")
                # 方法2：尝试使用 forward 方法
                elif hasattr(self.model, 'forward') and callable(getattr(self.model, 'forward')):
                    try:
                        forward_result = self.model.forward()
                        if isinstance(forward_result, tuple) and len(forward_result) == 2:
                            user_all_embeddings, item_all_embeddings = forward_result
                            self.logger.info("Using model.forward() to extract embeddings")
                        else:
                            raise ValueError("Forward output format not recognized")
                    except Exception as e:
                        self.logger.warning(f"Forward method failed: {e}, falling back to embedding layer")
                        raise
                else:
                    raise AttributeError("No suitable method found for extracting embeddings")

            except Exception as e:
                # 回退方案：使用原始 embedding 层
                self.logger.warning(f"Failed to get propagated embeddings: {e}")
                self.logger.warning("Falling back to raw embedding layer (layer 0)")
                user_all_embeddings = self.model.user_embedding.weight
                item_all_embeddings = self.model.item_embedding.weight

            # 添加L2归一化处理
            user_all_embeddings = F.normalize(user_all_embeddings, p=2, dim=1)
            item_all_embeddings = F.normalize(item_all_embeddings, p=2, dim=1)
            self.logger.info("Applied L2 normalization to user and item embeddings")

            # 验证 embedding 和 ID 映射的正确性
            self._validate_embedding_mapping(
                user_dict, item_dict,
                user_all_embeddings, item_all_embeddings
            )

            # Define default save path if not provided
            if save_path is None:
                save_path = './save_data'
                os.makedirs(save_path,exist_ok=True)
            else:
                os.makedirs(save_path,exist_ok=True)

            # Save embeddings to disk
            user_embedding_file = f"{save_path}/user_embeddings.csv"
            item_embedding_file = f"{save_path}/item_embeddings.csv"

            # 保存 user embeddings
            try:
                user_emb_dict = {user_id: user_all_embeddings[int(idx)].cpu().numpy() for user_id, idx in user_dict.items()}

                with open(user_embedding_file, 'w') as f:
                    for user_id, embed in user_emb_dict.items():
                        f.write(f"{user_id}\t" + " ".join(map(str, embed)) + "\n")

                # 成功写入后记录日志
                self.logger.info(f"User embeddings saved to {user_embedding_file} (count: {len(user_emb_dict)}, dim: {user_all_embeddings.shape[1]})")
            except Exception as e:
                self.logger.error(f"Failed to save user embeddings to {user_embedding_file}: {str(e)}")
                raise

            # 保存 item embeddings
            try:
                item_emb_dict = {item_id: item_all_embeddings[int(idx)].cpu().numpy() for item_id, idx in item_dict.items()}

                with open(item_embedding_file, 'w') as f:
                    for item_id, embed in item_emb_dict.items():
                        f.write(f"{item_id}\t" + " ".join(map(str, embed)) + "\n")

                # 成功写入后记录日志
                self.logger.info(f"Item embeddings saved to {item_embedding_file} (count: {len(item_emb_dict)}, dim: {item_all_embeddings.shape[1]})")
            except Exception as e:
                self.logger.error(f"Failed to save item embeddings to {item_embedding_file}: {str(e)}")
                raise


            # return {'user_embeddings': user_emb_dict, 'item_embeddings': item_emb_dict}


    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

