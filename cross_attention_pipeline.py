import bpe
import cross_attention_train
import cross_attention_samples

bpe.save_merges()
cross_attention_train.train()
cross_attention_samples.run()
