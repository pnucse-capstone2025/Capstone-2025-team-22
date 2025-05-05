import re
import argparse
import matplotlib.pyplot as plt
import os

def parse_log(log_path):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    train_pattern = re.compile(r"Step: (\d+), Loss: ([0-9\.]+)")
    total_val_pattern = re.compile(r"Total Step: \d+, Total Val loss: ([0-9\.]+), Acc: ([0-9\.]+)")

    with open(log_path, 'r') as f:
        i = 1
        for line in f:
            m_train = train_pattern.search(line)
            if m_train:
                train_steps.append(int(m_train.group(1)))
                train_losses.append(float(m_train.group(2)))
                continue
                
            m_total = total_val_pattern.search(line)
            if m_total:
                val_steps.append(i * 300)
                val_losses.append(float(m_total.group(1)))
                i += 1
                

    return train_steps, train_losses, val_steps, val_losses


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss from log file')
    parser.add_argument('--logfile', help='Path to the .log file')
    args = parser.parse_args()

    train_steps, train_losses, val_steps, val_losses = parse_log(args.logfile)

    plt.figure(figsize=(12, 6))
    
    # Plot validation loss first (averaged per epoch)
    if val_steps and len(val_steps) > 0:
        # Plot line connecting points (behind points)
        plt.plot(val_steps, val_losses, color='red', linestyle='-', zorder=1)
        # Plot points (on top of the line)
        plt.scatter(val_steps, val_losses, color='red', s=100, label='Validation Loss (avg per epoch)', zorder=2)
    
    # Plot training loss (all steps) - behind validation
    plt.plot(train_steps, train_losses, color='blue', label='Training Loss (per step)', alpha=0.7, zorder=0)
    
    # Annotate each validation point with its value
    if val_steps and len(val_steps) > 1:  # Need at least 2 points to mark the second one
        for i, (x, y) in enumerate(zip(val_steps, val_losses)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
    # Add special annotation for the minimum validation loss point
    min_index = val_losses.index(min(val_losses))
    best_x, best_y = val_steps[min_index], val_losses[min_index]
    plt.annotate('Best Model!', 
                xy=(best_x, best_y),
                xytext=(20, 40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # save to current directory
    save_path = os.path.join(os.getcwd(), 'training_vs_validation_loss.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == '__main__':
    main()
