import data
import lang_train
import lang_samples

data.save_lang_merges()
lang_train.train()
lang_samples.run()
