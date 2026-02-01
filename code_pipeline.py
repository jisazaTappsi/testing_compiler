import data
import time
import code_train
import code_samples
import data_generator

#data_generator.generate()
data.save_code_merges()
start = time.time()
last_val_loss = code_train.train()
end = time.time()
code_samples.run()
duration = end - start
print(f'It took: {round(duration)}s')
print(f'{last_val_loss=}')
print(f'time-complexity score: {1 / (last_val_loss*duration)}')
