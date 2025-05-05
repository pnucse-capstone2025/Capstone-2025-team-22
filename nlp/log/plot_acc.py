import re
import argparse
import matplotlib.pyplot as plt
import os

def parse_log(log_path):
    train_steps, train_accs = [], []
    val_steps, val_accs = [], []

    train_pattern = re.compile(r"Step: (\d+), Loss: [0-9\.]+, Acc: ([0-9\.]+)")
    total_val_pattern = re.compile(r"Total Step: \d+, Total Val loss: [0-9\.]+, Acc: ([0-9\.]+)")

    with open(log_path, 'r') as f:
        i = 1
        for line in f:
            m_train = train_pattern.search(line)
            if m_train:
                train_steps.append(int(m_train.group(1)))
                train_accs.append(float(m_train.group(2)) * 100)  # Convert to percentage
                continue
                
            m_total = total_val_pattern.search(line)
            if m_total:
                val_steps.append(i * 300)
                val_accs.append(float(m_total.group(1)) * 100)  # Convert to percentage
                i += 1

    return train_steps, train_accs, val_steps, val_accs


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation accuracy from log file')
    parser.add_argument('--logfile', help='Path to the .log file')
    args = parser.parse_args()

    train_steps, train_accs, val_steps, val_accs = parse_log(args.logfile)

    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy (all steps)
    plt.plot(train_steps, train_accs, color='blue', label='Training Accuracy (per step)', alpha=0.7, zorder=0)
    
    # Plot validation accuracy (averaged per epoch)
    if val_steps and len(val_steps) > 0:
        # Plot line connecting points (behind points)
        plt.plot(val_steps, val_accs, color='red', linestyle='-', zorder=1)
        # Plot points (on top of the line)
        plt.scatter(val_steps, val_accs, color='red', s=100, label='Validation Accuracy (per epoch)', zorder=2)
    
    # Annotate each validation point with its value
    if val_steps and len(val_steps) > 1:
        for i, (x, y) in enumerate(zip(val_steps, val_accs)):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
    # Add special annotation for the maximum validation accuracy point
    if val_accs:  # Check if val_accs is not empty
        max_index = val_accs.index(max(val_accs))
        best_x, best_y = val_steps[max_index], val_accs[max_index]
        plt.annotate('Best Model!', 
                    xy=(best_x, best_y),
                    xytext=(20, -40),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(os.getcwd(), 'training_vs_validation_accuracy.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
