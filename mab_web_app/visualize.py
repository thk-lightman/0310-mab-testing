import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('visualize')

class MABVisualizer:
    def __init__(self, data_dir='data', output_dir='static/images'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        sns.set(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
    
    def load_mab_data(self, report_file='mab_report.json'):
        """Load MAB report data"""
        file_path = os.path.join(self.data_dir, report_file)
        try:
            with open(file_path, 'r') as f:
                report_data = json.load(f)
            
            logger.info(f"Loaded MAB report from {file_path}")
            return report_data
        except Exception as e:
            logger.error(f"Error loading MAB report: {str(e)}")
            return None
    
    def load_interaction_data(self, data_file='interactions.json'):
        """Load interaction data from JSON file"""
        file_path = os.path.join(self.data_dir, data_file)
        try:
            with open(file_path, 'r') as f:
                interaction_data = json.load(f)
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(interaction_data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            logger.info(f"Loaded interaction data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading interaction data: {str(e)}")
            return None
    
    def _save_plot(self, filename):
        """Helper function to save plot to output directory"""
        file_path = os.path.join(self.output_dir, filename)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {file_path}")
        return file_path
    
    def plot_conversion_rates(self, report_data=None, save=True):
        """Plot current conversion rates for each design"""
        if report_data is None:
            report_data = self.load_mab_data()
        
        if not report_data:
            logger.error("No report data available for plotting conversion rates")
            return None
        
        designs = report_data['designs']
        names = [d['name'] for d in designs]
        rates = [d['conversion_rate'] for d in designs]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, rates, color=sns.color_palette("husl", len(names)))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom')
        
        plt.title(f"Conversion Rates by Design ({report_data['algorithm']} Algorithm)")
        plt.xlabel("Design Version")
        plt.ylabel("Conversion Rate")
        plt.ylim(0, max(rates) * 1.2)  # Add some space for labels
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_plot(f"conversion_rates_{timestamp}.png")
        else:
            plt.show()
    
    def plot_impression_distribution(self, report_data=None, save=True):
        """Plot distribution of impressions among designs"""
        if report_data is None:
            report_data = self.load_mab_data()
        
        if not report_data:
            logger.error("No report data available for plotting impression distribution")
            return None
        
        designs = report_data['designs']
        names = [d['name'] for d in designs]
        impressions = [d['impressions'] for d in designs]
        
        plt.figure(figsize=(10, 6))
        plt.pie(impressions, labels=names, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette("husl", len(names)))
        plt.axis('equal')  
        plt.title(f"Impression Distribution ({report_data['algorithm']} Algorithm)")
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_plot(f"impression_distribution_{timestamp}.png")
        else:
            plt.show()
    
    def plot_cumulative_rewards(self, interaction_data=None, save=True):
        """Plot cumulative rewards (conversions) over time"""
        if interaction_data is None:
            interaction_data = self.load_interaction_data()
        
        if interaction_data is None or len(interaction_data) == 0:
            logger.error("No interaction data available for plotting cumulative rewards")
            return None
        
        # Filter only successful conversions
        conversions = interaction_data[interaction_data['success'] == True].copy()
        
        if len(conversions) == 0:
            logger.warning("No successful conversions found in interaction data")
            return None
        
        # Group by design and timestamp, get cumulative count
        designs = conversions['design_version'].unique()
        
        plt.figure(figsize=(12, 8))
        
        for design in designs:
            design_data = conversions[conversions['design_version'] == design]
            design_data = design_data.sort_values('timestamp')
            
            # Create cumulative count
            cumulative = np.arange(1, len(design_data) + 1)
            
            plt.plot(design_data['timestamp'], cumulative, 
                    marker='o', markersize=4, linestyle='-', 
                    label=f"Design {design}")
        
        plt.title("Cumulative Conversions Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Conversions")
        plt.legend()
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_plot(f"cumulative_conversions_{timestamp}.png")
        else:
            plt.show()
    
    def plot_conversion_rate_evolution(self, interaction_data=None, window_size=20, save=True):
        """Plot evolution of conversion rates over time with moving average"""
        if interaction_data is None:
            interaction_data = self.load_interaction_data()
        
        if interaction_data is None or len(interaction_data) == 0:
            logger.error("No interaction data available for plotting conversion rate evolution")
            return None
        
        # Get unique designs
        designs = interaction_data['design_version'].unique()
        
        plt.figure(figsize=(12, 8))
        
        for design in designs:
            # Filter data for this design
            design_data = interaction_data[interaction_data['design_version'] == design].copy()
            
            if len(design_data) < window_size:
                logger.warning(f"Not enough data for Design {design} to calculate moving average")
                continue
            
            # Sort by timestamp
            design_data = design_data.sort_values('timestamp')
            
            # Create binary success column (1 for success, 0 for failure)
            design_data['binary_success'] = design_data['success'].astype(int)
            
            # Calculate moving average
            design_data['moving_avg'] = design_data['binary_success'].rolling(window=window_size).mean()
            
            # Plot
            plt.plot(design_data['timestamp'], design_data['moving_avg'], 
                    label=f"Design {design}", linewidth=2)
        
        plt.title(f"Conversion Rate Evolution (Moving Avg Window={window_size})")
        plt.xlabel("Time")
        plt.ylabel("Conversion Rate (Moving Average)")
        plt.legend()
        plt.grid(True)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._save_plot(f"conversion_rate_evolution_{timestamp}.png")
        else:
            plt.show()
    
    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        logger.info("Generating all MAB visualizations")
        
        # Load data
        report_data = self.load_mab_data()
        interaction_data = self.load_interaction_data()
        
        # Generate plots
        plots = []
        
        if report_data:
            plots.append(self.plot_conversion_rates(report_data))
            plots.append(self.plot_impression_distribution(report_data))
        
        if interaction_data is not None and len(interaction_data) > 0:
            plots.append(self.plot_cumulative_rewards(interaction_data))
            plots.append(self.plot_conversion_rate_evolution(interaction_data))
        
        return [p for p in plots if p]


if __name__ == "__main__":
    visualizer = MABVisualizer()
    visualizer.generate_all_visualizations() 