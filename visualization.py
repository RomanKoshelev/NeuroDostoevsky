import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def running_mean(arr, num):
    if len(arr) == 0:
        return [0]
    cumsum = np.cumsum(np.insert(arr, 0, [arr[0]]*num))
    return (cumsum[num:] - cumsum[:-num]) / num 


def show_train_stats(ep, tr_step, tr_losses, mean_win):
    plt.figure(figsize=(16,12))
    fontsize = 14

    # Loss
    plt.subplot(221)
    tr_means = running_mean(tr_losses, mean_win)
    tr_loss = tr_means[-1]
    plt.title("Epoch %.2f | Step %d | Train loss %.2e" % (ep, tr_step, tr_loss), fontsize=fontsize)
    plt.plot(tr_losses, 'c')
    tr, = plt.plot(tr_means, 'b', label="Train")
    plt.legend(handles=[tr])
    plt.grid(True)
    plt.xlim(xmin=0)
   
    # Show
    clear_output(True)    
    plt.show()

    
def show_train_stats_ex(ep, step, tr_losses, va_losses, mean_win=30, log_scale=False):
    plt.figure(figsize=(16,12))
    fontsize = 14

    # Loss
    plt.subplot(221)
    tr_means = running_mean(tr_losses, mean_win)
    va_means = running_mean(va_losses, mean_win)
    tr_loss = tr_means[-1]
    va_loss = va_means[-1]
    plt.title("Epoch %.2f | Step %d | Valid %.2e | Train %.2e" % (ep, step, va_loss, tr_loss), fontsize=fontsize)
    if not log_scale:
        plt.plot(tr_losses, 'c')
        va, = plt.plot(va_means, 'r', label="Valid")
        tr, = plt.plot(tr_means, 'b', label="Train")
        plt.legend(handles=[va, tr])
    else:
        plt.yscale("log")
        tr, = plt.plot(tr_means, 'b', label="Train")
        plt.legend(handles=[tr])
    plt.grid(True)
    plt.xlim(xmin=0)
 
    # Show
    clear_output(True)    
    plt.show()