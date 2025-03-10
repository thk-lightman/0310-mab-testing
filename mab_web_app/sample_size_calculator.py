"""
Sample Size Calculator for MAB Testing.
This module provides tools to calculate the required sample size for MAB tests
to achieve statistical significance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from datetime import datetime
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sample_size.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sample_size')

class MABSampleSizeCalculator:
    """
    Calculator for determining the appropriate sample size for MAB tests.
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the sample size calculator.
        
        Args:
            output_dir (str): Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        sns.set(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
    
    def calculate_sample_size(self, baseline_conversion_rate, minimum_detectable_effect,
                             alpha=0.05, power=0.8, n_variants=3):
        """
        Calculate required sample size for MAB test.
        
        Args:
            baseline_conversion_rate (float): Expected baseline conversion rate (0-1)
            minimum_detectable_effect (float): Smallest meaningful change to detect (0-1)
            alpha (float): Significance level (Type I error probability)
            power (float): Power (1 - Type II error probability)
            n_variants (int): Number of variants in the test
        
        Returns:
            dict: Sample size information
        """
        # Validate inputs
        if not 0 < baseline_conversion_rate < 1:
            raise ValueError("Baseline conversion rate must be between 0 and 1")
        if not 0 < minimum_detectable_effect < 1:
            raise ValueError("Minimum detectable effect must be between 0 and 1")
        
        # Calculate expected improved conversion rate
        improved_rate = baseline_conversion_rate * (1 + minimum_detectable_effect)
        
        # Calculate standard deviation under null and alternative hypotheses
        sd1 = np.sqrt(baseline_conversion_rate * (1 - baseline_conversion_rate))
        sd2 = np.sqrt(improved_rate * (1 - improved_rate))
        
        # Adjusted significance level for multiple comparisons (Bonferroni correction)
        adjusted_alpha = alpha / (n_variants - 1)
        
        # Calculate z-values
        z_alpha = stats.norm.ppf(1 - adjusted_alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Calculate sample size per variant
        numerator = (z_alpha * np.sqrt(2 * sd1**2) + z_beta * np.sqrt(sd1**2 + sd2**2))**2
        denominator = (improved_rate - baseline_conversion_rate)**2
        sample_size_per_variant = np.ceil(numerator / denominator)
        
        # Calculate total sample size
        total_sample_size = sample_size_per_variant * n_variants
        
        # Expected effect size (Cohen's h)
        cohens_h = 2 * np.arcsin(np.sqrt(improved_rate)) - 2 * np.arcsin(np.sqrt(baseline_conversion_rate))
        
        # Prepare result
        result = {
            'baseline_conversion_rate': baseline_conversion_rate,
            'improved_conversion_rate': improved_rate,
            'minimum_detectable_effect': minimum_detectable_effect,
            'alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'power': power,
            'variants': n_variants,
            'cohens_h': cohens_h,
            'sample_size_per_variant': int(sample_size_per_variant),
            'total_sample_size': int(total_sample_size),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Sample size calculation: {result['total_sample_size']} total samples needed")
        return result
    
    def save_calculation(self, result, file_name='sample_size_calculation.json'):
        """
        Save sample size calculation to file.
        
        Args:
            result (dict): Sample size calculation result
            file_name (str): Name of output file
        """
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved sample size calculation to {file_path}")
        
        return file_path
    
    def plot_sample_size_vs_mde(self, baseline_rate, mde_range, alpha=0.05, power=0.8, n_variants=3, 
                               save=True, file_name='sample_size_vs_mde.png'):
        """
        Plot required sample size vs. minimum detectable effect.
        
        Args:
            baseline_rate (float): Baseline conversion rate
            mde_range (list): Range of minimum detectable effects to plot
            alpha (float): Significance level
            power (float): Statistical power
            n_variants (int): Number of variants
            save (bool): Whether to save the plot
            file_name (str): Name of output file
        
        Returns:
            str: Path to saved file if save=True
        """
        sample_sizes = []
        
        for mde in mde_range:
            result = self.calculate_sample_size(
                baseline_rate, mde, alpha, power, n_variants
            )
            sample_sizes.append(result['total_sample_size'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(mde_range, sample_sizes, 'o-', linewidth=2)
        plt.title(f'Required Sample Size vs. Minimum Detectable Effect\n(Baseline Rate: {baseline_rate:.1%}, {n_variants} Variants)')
        plt.xlabel('Minimum Detectable Effect')
        plt.ylabel('Total Sample Size')
        plt.grid(True)
        plt.ticklabel_format(style='plain', axis='y')
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        if save:
            file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {file_path}")
            return file_path
        else:
            plt.show()
            return None
    
    def plot_sample_size_vs_baseline(self, baseline_range, mde, alpha=0.05, power=0.8, n_variants=3,
                                   save=True, file_name='sample_size_vs_baseline.png'):
        """
        Plot required sample size vs. baseline conversion rate.
        
        Args:
            baseline_range (list): Range of baseline conversion rates to plot
            mde (float): Minimum detectable effect
            alpha (float): Significance level
            power (float): Statistical power
            n_variants (int): Number of variants
            save (bool): Whether to save the plot
            file_name (str): Name of output file
        
        Returns:
            str: Path to saved file if save=True
        """
        sample_sizes = []
        
        for rate in baseline_range:
            result = self.calculate_sample_size(
                rate, mde, alpha, power, n_variants
            )
            sample_sizes.append(result['total_sample_size'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_range, sample_sizes, 'o-', linewidth=2)
        plt.title(f'Required Sample Size vs. Baseline Conversion Rate\n(MDE: {mde:.1%}, {n_variants} Variants)')
        plt.xlabel('Baseline Conversion Rate')
        plt.ylabel('Total Sample Size')
        plt.grid(True)
        plt.ticklabel_format(style='plain', axis='y')
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        if save:
            file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {file_path}")
            return file_path
        else:
            plt.show()
            return None
    
    def calculate_test_duration(self, sample_size, daily_visitors, percentage_in_test=1.0):
        """
        Calculate the expected duration of the test in days.
        
        Args:
            sample_size (int): Required sample size
            daily_visitors (int): Expected daily visitors
            percentage_in_test (float): Percentage of visitors included in the test (0-1)
        
        Returns:
            float: Expected test duration in days
        """
        if percentage_in_test <= 0 or percentage_in_test > 1:
            raise ValueError("Percentage in test must be between 0 and 1")
        
        daily_test_visitors = daily_visitors * percentage_in_test
        duration_days = sample_size / daily_test_visitors
        
        return duration_days
    
    def generate_report(self, baseline_rate, mde, daily_visitors, percentage_in_test=1.0,
                       alpha=0.05, power=0.8, n_variants=3, output_file='mab_test_plan.json'):
        """
        Generate a comprehensive MAB test plan report.
        
        Args:
            baseline_rate (float): Baseline conversion rate
            mde (float): Minimum detectable effect
            daily_visitors (int): Expected daily visitors
            percentage_in_test (float): Percentage of visitors in test
            alpha (float): Significance level
            power (float): Statistical power
            n_variants (int): Number of variants
            output_file (str): Output file name
        
        Returns:
            dict: Test plan report
        """
        # Calculate sample size
        sample_size_result = self.calculate_sample_size(
            baseline_rate, mde, alpha, power, n_variants
        )
        
        # Calculate test duration
        duration_days = self.calculate_test_duration(
            sample_size_result['total_sample_size'], 
            daily_visitors, 
            percentage_in_test
        )
        
        # Generate plots
        mde_range = np.linspace(0.01, 0.2, 20)
        baseline_range = np.linspace(0.01, 0.1, 10)
        
        mde_plot = self.plot_sample_size_vs_mde(
            baseline_rate, mde_range, alpha, power, n_variants, 
            save=True, file_name='sample_size_vs_mde.png'
        )
        
        baseline_plot = self.plot_sample_size_vs_baseline(
            baseline_range, mde, alpha, power, n_variants,
            save=True, file_name='sample_size_vs_baseline.png'
        )
        
        # Compile report
        report = {
            'test_parameters': {
                'baseline_conversion_rate': baseline_rate,
                'minimum_detectable_effect': mde,
                'significance_level': alpha,
                'power': power,
                'variants': n_variants,
                'daily_visitors': daily_visitors,
                'percentage_in_test': percentage_in_test
            },
            'sample_size': sample_size_result,
            'test_duration': {
                'days': duration_days,
                'weeks': duration_days / 7,
                'months': duration_days / 30
            },
            'plots': {
                'sample_size_vs_mde': os.path.basename(mde_plot) if mde_plot else None,
                'sample_size_vs_baseline': os.path.basename(baseline_plot) if baseline_plot else None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        file_path = os.path.join(self.output_dir, output_file)
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated MAB test plan and saved to {file_path}")
        return report

if __name__ == '__main__':
    # Example usage
    calculator = MABSampleSizeCalculator()
    
    # Generate a test plan for a hypothetical website
    test_plan = calculator.generate_report(
        baseline_rate=0.05,  # 5% baseline conversion rate
        mde=0.10,            # Detect a 10% relative improvement
        daily_visitors=1000, # 1000 visitors per day
        percentage_in_test=0.5,  # 50% of traffic included in test
        n_variants=3         # 3 design variants
    )
    
    # Print summary
    print("\nMAB Test Plan Summary:")
    print(f"Required sample size: {test_plan['sample_size']['total_sample_size']} visitors")
    print(f"Expected test duration: {test_plan['test_duration']['days']:.1f} days")
    print(f"Expected conversions: {test_plan['sample_size']['total_sample_size'] * test_plan['test_parameters']['baseline_conversion_rate']:.0f}")
    print(f"Report saved to: {os.path.join(calculator.output_dir, 'mab_test_plan.json')}") 