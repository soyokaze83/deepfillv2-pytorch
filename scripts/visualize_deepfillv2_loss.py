"""
DeepFillv2 Training Log Parser and Loss Visualizer

This script parses training logs from DeepFillv2 inpainting model,
extracts loss values, and creates visualizations.

Usage:
    python visualize_deepfillv2_loss.py <log_file_path> [--output_dir <output_directory>]

Example:
    python visualize_deepfillv2_loss.py train_logs.txt --output_dir ./plots
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_training_logs(log_file_path: str) -> dict:
    """
    Parse DeepFillv2 training logs and extract loss values.
    
    Args:
        log_file_path: Path to the training log file
        
    Returns:
        Dictionary containing iterations and loss values for each loss type
    """
    losses = {
        'iterations': [],
        'd_loss': [],      # Discriminator loss
        'g_loss': [],      # Generator loss
        'ae_loss': [],     # Total autoencoder loss
        'ae_loss1': [],    # Autoencoder loss component 1
        'ae_loss2': [],    # Autoencoder loss component 2
    }
    
    # Regex patterns for parsing
    iter_pattern = re.compile(r'@iter:\s*(\d+):')
    loss_patterns = {
        'd_loss': re.compile(r'd_loss:\s*([-\d.]+)'),
        'g_loss': re.compile(r'g_loss:\s*([-\d.]+)'),
        'ae_loss': re.compile(r'ae_loss:\s*([-\d.]+)'),
        'ae_loss1': re.compile(r'ae_loss1:\s*([-\d.]+)'),
        'ae_loss2': re.compile(r'ae_loss2:\s*([-\d.]+)'),
    }
    
    current_iter = None
    current_losses = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for iteration line
            iter_match = iter_pattern.search(line)
            if iter_match:
                # Save previous iteration's losses if complete
                if current_iter is not None and len(current_losses) == 5:
                    losses['iterations'].append(current_iter)
                    for key in loss_patterns.keys():
                        losses[key].append(current_losses.get(key, np.nan))
                
                current_iter = int(iter_match.group(1))
                current_losses = {}
                continue
            
            # Check for loss values
            for loss_name, pattern in loss_patterns.items():
                match = pattern.search(line)
                if match:
                    try:
                        current_losses[loss_name] = float(match.group(1))
                    except ValueError:
                        pass
                    break
        
        # Don't forget the last iteration
        if current_iter is not None and len(current_losses) == 5:
            losses['iterations'].append(current_iter)
            for key in loss_patterns.keys():
                losses[key].append(current_losses.get(key, np.nan))
    
    # Convert to numpy arrays
    for key in losses:
        losses[key] = np.array(losses[key])
    
    return losses


def plot_combined_losses(losses: dict, output_path: str = None, figsize: tuple = (14, 8)):
    """
    Create a single plot with all losses overlaid.
    
    Args:
        losses: Dictionary containing parsed loss data
        output_path: Path to save the figure (optional)
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = losses['iterations']
    
    # Define colors and labels for each loss
    loss_config = {
        'd_loss': {'color': '#e74c3c', 'label': 'Discriminator Loss (d_loss)', 'alpha': 0.9},
        'g_loss': {'color': '#3498db', 'label': 'Generator Loss (g_loss)', 'alpha': 0.9},
        'ae_loss': {'color': '#2ecc71', 'label': 'AE Loss Total (ae_loss)', 'alpha': 0.9},
        'ae_loss1': {'color': '#9b59b6', 'label': 'AE Loss 1 (ae_loss1)', 'alpha': 0.7},
        'ae_loss2': {'color': '#f39c12', 'label': 'AE Loss 2 (ae_loss2)', 'alpha': 0.7},
    }
    
    for loss_name, config in loss_config.items():
        ax.plot(iterations, losses[loss_name], 
                color=config['color'], 
                label=config['label'],
                alpha=config['alpha'],
                linewidth=1.5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('DeepFillv2 Training Losses', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(iterations.min(), iterations.max())
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Combined plot saved to: {output_path}")
    
    return fig, ax


def plot_individual_losses(losses: dict, output_path: str = None, figsize: tuple = (16, 12)):
    """
    Create subplots for each individual loss component.
    
    Args:
        losses: Dictionary containing parsed loss data
        output_path: Path to save the figure (optional)
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    iterations = losses['iterations']
    
    # Configuration for each subplot
    subplot_config = [
        ('d_loss', 'Discriminator Loss', '#e74c3c'),
        ('g_loss', 'Generator Loss', '#3498db'),
        ('ae_loss', 'AE Loss (Total)', '#2ecc71'),
        ('ae_loss1', 'AE Loss 1 (Coarse)', '#9b59b6'),
        ('ae_loss2', 'AE Loss 2 (Refined)', '#f39c12'),
    ]
    
    for idx, (loss_name, title, color) in enumerate(subplot_config):
        ax = axes[idx]
        loss_values = losses[loss_name]
        
        # Main line plot
        ax.plot(iterations, loss_values, color=color, linewidth=1.2, alpha=0.8)
        
        # Add smoothed trend line (moving average)
        window_size = min(50, len(loss_values) // 10) if len(loss_values) > 100 else 5
        if window_size > 1:
            smoothed = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
            smoothed_iters = iterations[window_size-1:]
            ax.plot(smoothed_iters, smoothed, color='black', linewidth=2, 
                   alpha=0.7, linestyle='--', label=f'MA({window_size})')
            ax.legend(loc='upper right', fontsize=9)
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(iterations.min(), iterations.max())
        
        # Add statistics annotation
        stats_text = f'Final: {loss_values[-1]:.4f}\nMin: {loss_values.min():.4f}\nMax: {loss_values.max():.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide the unused subplot (6th position)
    axes[5].axis('off')
    
    # Add summary statistics to the empty subplot area
    ax_summary = axes[5]
    ax_summary.axis('off')
    
    summary_text = "Training Summary\n" + "="*30 + "\n\n"
    summary_text += f"Total Iterations: {int(iterations.max())}\n"
    summary_text += f"Data Points: {len(iterations)}\n\n"
    summary_text += "Final Loss Values:\n"
    summary_text += f"  • D Loss: {losses['d_loss'][-1]:.4f}\n"
    summary_text += f"  • G Loss: {losses['g_loss'][-1]:.4f}\n"
    summary_text += f"  • AE Loss: {losses['ae_loss'][-1]:.4f}\n"
    summary_text += f"  • AE Loss1: {losses['ae_loss1'][-1]:.4f}\n"
    summary_text += f"  • AE Loss2: {losses['ae_loss2'][-1]:.4f}\n"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('DeepFillv2 Training Losses - Individual Components', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Individual plots saved to: {output_path}")
    
    return fig, axes


def print_summary(losses: dict):
    """Print a summary of the parsed training data."""
    print("\n" + "="*60)
    print("DeepFillv2 Training Log Summary")
    print("="*60)
    print(f"\nTotal iterations parsed: {len(losses['iterations'])}")
    print(f"Iteration range: {int(losses['iterations'].min())} - {int(losses['iterations'].max())}")
    print("\nLoss Statistics:")
    print("-"*60)
    print(f"{'Loss Type':<15} {'Min':>12} {'Max':>12} {'Final':>12} {'Mean':>12}")
    print("-"*60)
    
    for loss_name in ['d_loss', 'g_loss', 'ae_loss', 'ae_loss1', 'ae_loss2']:
        values = losses[loss_name]
        print(f"{loss_name:<15} {values.min():>12.4f} {values.max():>12.4f} "
              f"{values[-1]:>12.4f} {values.mean():>12.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parse and visualize DeepFillv2 training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_deepfillv2_loss.py train_logs.txt
    python visualize_deepfillv2_loss.py train_logs.txt --output_dir ./plots
    python visualize_deepfillv2_loss.py train_logs.txt --no-show
        """
    )
    parser.add_argument('log_file', type=str, help='Path to the training log file')
    parser.add_argument('--output_dir', type=str, default='.', 
                       help='Directory to save output plots (default: current directory)')
    parser.add_argument('--no-show', action='store_true', 
                       help='Do not display plots (only save)')
    parser.add_argument('--format', type=str, default='png', 
                       choices=['png', 'pdf', 'svg'],
                       help='Output format for saved plots (default: png)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the logs
    print(f"Parsing training logs from: {args.log_file}")
    losses = parse_training_logs(args.log_file)
    
    if len(losses['iterations']) == 0:
        print("Error: No training data found in the log file!")
        return 1
    
    # Print summary
    print_summary(losses)
    
    # Generate plots
    combined_path = output_dir / f"deepfillv2_combined_losses.{args.format}"
    individual_path = output_dir / f"deepfillv2_individual_losses.{args.format}"
    
    plot_combined_losses(losses, output_path=str(combined_path))
    plot_individual_losses(losses, output_path=str(individual_path))
    
    if not args.no_show:
        plt.show()
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
