import math

def anneal_beta(epoch, beta, total_epochs=120, start_epoch=29, max_val=0.1):
            if epoch < start_epoch:
                return beta
            else:
                progress = (epoch - start_epoch) / (total_epochs - start_epoch)
                return max_val / (1 + math.exp(-12 * (progress - 0.5)))


start_beta = 0.001


for i in range(150):
    print(start_beta)
    start_beta = anneal_beta(i, start_beta)
