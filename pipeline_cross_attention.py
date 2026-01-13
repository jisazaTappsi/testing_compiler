import bpa
import train_cross_attention
import samples_cross_attention

bpa.save_merges()
train_cross_attention.train()
samples_cross_attention.run()
