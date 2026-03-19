# coding: utf-8

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str,load_csv_to_dict
import platform, os
import csv


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    # 清除旧的日志文件和模型权重
    os.system('rm -rf  ./ImRec/log/*')
    os.system('rm -rf  ./ImRec/saved/*')
    os.system('rm -rf  ./ImRec/save_data/*')
    os.system('rm -rf  ./ImRec/recommend_topk/*')
    # os.system('pip freeze > requirements.txt')
    #获取训练数据
    os.system("rm -rf  ./ImRec/data/*")
 
 
 
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)
    
    os.system('rm -rf  ./ImRec/preprocessed_data/*')

    # load data
    dataset = RecDataset(config)

    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split(config['split_ratio'])
    logger.info('\n{}'.format(len(valid_dataset)))
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))


    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    
    # combinations 修改GCF的超参数配置，减少无用的参数组合
    combinators = list(product(*hyper_ls))[:7]
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # random seed reset
        init_seed(config['seed'])

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        if idx==0:
            logger.info(model)
        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=save_model, is_test=True, idx=idx)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, test_result))

        # save all user/item embedding 
        user_dict_path = os.path.join(dataset.preprocessed_dataset_path,
                                     '{}_u_{}_mapping.csv'.format(dataset.dataset_name, dataset.ui_core_splitting_str))
        item_dict_path = os.path.join(dataset.preprocessed_dataset_path,
                                     '{}_i_{}_mapping.csv'.format(dataset.dataset_name, dataset.ui_core_splitting_str))
        save_path = ' ./ImRec/inter_embs'
        user_dict = load_csv_to_dict(user_dict_path)
        item_dict = load_csv_to_dict(item_dict_path)
        trainer.save_all_embedding(user_dict,item_dict,load_best_model=save_model,save_path=save_path,idx=idx)


        # save best test
        if test_result[val_metric] > best_test_value:
            best_test_value = test_result[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(test_result)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'], p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

