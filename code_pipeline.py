import data
import time
import code_train
import code_samples
import data_generator

data_generator.generate()
data.save_code_merges()
start = time.time()
last_val_loss = code_train.train()
end = time.time()
computation_percentage = code_samples.run(num_samples=500)
duration_minutes = (end - start) / 60
print(f'It took: {round(duration_minutes, 1)} min')
print(f'{last_val_loss=}')
print(f'time-complexity score: {1 / (last_val_loss * duration_minutes)}')
print(f'time-complexity computation score: { computation_percentage / duration_minutes }')
