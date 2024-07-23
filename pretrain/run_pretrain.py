from config import args
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH
from trainer import BERT4ETHTrainer
import pickle as pkl
from vocab import FreqVocab
from utils import save_log, parameters_log
import datetime
import os
def train():
    # create log
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log = open(args.log_dir + 'pretrain_' + current_time + '.txt', 'a')
    save_log(log, f"========== Start of Pretrain Log - {current_time} ==========\n")
    parameters_log(log, args)
    # prepare dataset
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = BERT4ETH(args)

    # trainer
    trainer = BERT4ETHTrainer(args, vocab, model, train_loader)
    trainer.train(log=log)

    save_log(log, f"========== End of Pretrain Log ==========\n")
    log.close()

if __name__ == '__main__':
    train()

